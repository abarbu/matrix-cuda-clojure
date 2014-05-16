(ns matrix-cuda-clojure.core
 (:require [clojure.core.matrix           :as m]
           [clojure.core.matrix.operators :as mo]
           [clojure.core.matrix.protocols :as mp]
           [clojure.core.matrix.implementations :as imp]
           [clojure.core.matrix.compliance-tester :as mc]))

(use '[sanity core improvements reader])
(use 'common-clojure.core)
(use 'clojure.reflect)

(import 'jcuda.runtime.JCuda 'jcuda.jcublas.JCublas 'jcuda.jcublas.JCublas2
        'jcuda.Sizeof 'jcuda.runtime.cudaMemcpyKind 'jcuda.jcublas.cublasHandle
        'jcuda.jcublas.cublasOperation 'jcuda.driver.CUmodule
        'jcuda.driver.JCudaDriver 'jcuda.driver.CUresult
        'jcuda.driver.CUdevice 'jcuda.driver.CUcontext 'jcuda.driver.CUfunction 'jcuda.driver.CUdeviceptr
        'jcuda.runtime.cudaDeviceProp)

(defn- ^jcuda.Pointer pointer+ [^jcuda.Pointer pointer ^Long offset] (.withByteOffset pointer offset))

;; One-time cuda initialization; TODO this really should be cleaned up

;;; Low-level CUDA access so that we can run kernels
(defrecord cuda-context [device context kernels-module blas])

;; This is provided because one day we will correctly honor it
;; It's a constant for now
(define *default-cuda-device* 0)

(define (initialize-cuda-context device-index)
 (let ((device (CUdevice.))
       (context (CUcontext.))
       (kernels-module (CUmodule.))
       (blas (cublasHandle.)))
  (JCudaDriver/setExceptionsEnabled true)
  (JCudaDriver/cuInit 0)
  (JCudaDriver/cuDeviceGet device device-index)
  (JCudaDriver/cuCtxCreate context 0 device)
  (JCudaDriver/cuModuleLoad kernels-module (resource-path "kernels.ptx"))
  (JCublas/setExceptionsEnabled true)
  (JCublas/cublasInit)
  (JCublas2/cublasCreate blas)
  (->cuda-context device context kernels-module blas)))

(define (cuda-device-properties)
 ;; TODO Should maybe do something smarter, 'name' sort of sucks
 (into {} (let ((properties (cudaDeviceProp.)))
           (JCuda/cudaGetDeviceProperties properties *default-cuda-device*)
           (map (lambda (a)
                 [(keyword (:name a))
                  (let ((i (clojure.lang.Reflector/getInstanceField
                            properties
                            (str (:name a)))))
                   (if (number? i) i (into [] i)))])
                (remove-if-not (lambda (a) (= (type a) clojure.reflect.Field))
                               (:members (reflect properties)))))))

(define *cuda-default-device-properties* (cuda-device-properties))

;; TODO Initialization
(define *default-cuda-context* (initialize-cuda-context *default-cuda-device*))

(define (thread-has-cuda-context?)
 (let ((c (CUcontext.)))
  (= (JCudaDriver/cuCtxGetCurrent c) CUresult/CUDA_SUCCESS)))

;; This is required in the repl because cider/nrepl can switch OS threads on us
(defn with-cuda-context
 ([f] (JCudaDriver/cuCtxSetCurrent (:context *default-cuda-context*)) (f))
 ([context f] (JCudaDriver/cuCtxSetCurrent context) (f)))

(declare cuda-launch-kernel)

;;; Pools

;; We have to manage the GPU memory we're going to allocate for our
;; core.matrix vectors and matrices. There are a few options:
;;     - finalizers
;;     - weak/phantom references
;;     - manual management
;;     - memory pools
;;
;; Finalizers can slow GC down significantly. Finalizers and
;; weak/phantom references are difficult to use because we can't be
;; sure when they're going to run. Calling System/gc in order to make
;; sure that we're going to finalize is just impractical because it
;; stops the world for a very long time even with modestly-sized
;; heaps. We also aren't able to track our memory usage.
;; 
;; Manual management isn't an option because it isn't compatible with
;; core.matrix. Every temporary matrix would need to be freed
;; explicitly. It's also just a pain.
;;
;; We're then left with memory pools which solve all of the problems
;; above at the expense of a small amount of occasional memory
;; management. Memory is by default allocated from a pool dynamic
;; variable *cuda-memory-pool*. A pool is a hash map of pointers to
;; device (CUDA) memory to information about those pointers, currently
;; a single integer the size of the memory. We record the size so that
;; we can quickly compute the total size of a pool. You can create
;; custom pools with (cuda-create-pool). All operations on pools are
;; thread-safe.
;;
;; Allocations are automatically added to the current global pool, but
;; you can record your own CUDA memory using (cuda-add-to-pool!
;; pointer size) and allocate directly to a pool using
;; (cuda-malloc-in-pool pointer size). (cuda-pool-size) returns the
;; total size, in bytes, of a pool. You can free pools with
;; (cuda-free-pool). All operations on single pools take an optional
;; last argument, a custom pool to use instead of the default
;; *cuda-memory. Adding to a pool is O(1), computing the size of a
;; pool and freeing the pool are O(size of pool).
;;
;; Often results will be computed in one pool but you want to hang on
;; to them while freeing all of the temporaries created during those
;; operations. You can move a pointer to a new pool allowing you to
;; safely free the previous pool with (cuda-move-between-pools!
;; pointer from-pool to-pool). This is a fast operation that can be
;; used regularly as it just does some housekeeping without performing
;; any CUDA operations and is O(1).

;; TODO A with-custom-pool operation that moves all desired pointers
;; into the current pool freeing all temporaries.

(define (cuda-create-pool) (ref {}))

;; The default memory pool
(def ^:dynamic *cuda-memory-pool* (cuda-create-pool))

;; This is used internally to store objects that should be pinned
;; Currently only used for the representative core.matrix object
(def ^:dynamic *cuda-permanent-pool* (cuda-create-pool))

(define (cuda-add-to-pool! pointer size & pool)
 (dosync (alter (if pool (first pool) *cuda-memory-pool*)
                #(merge % {pointer size}))))

(define (cuda-remove-from-pool! pointer & pool)
 (dosync (alter (if pool (first pool) *cuda-memory-pool*)
                #(dissoc % pointer))))

(define (cuda-pool-size . pool)
 (sum (map second (into [] (if pool @(first pool) @*cuda-memory-pool*)))))

(define (cuda-move-between-pools! pointer from-pool to-pool)
 (dosync (let ((size (get @from-pool pointer)))
          (alter from-pool #(dissoc % pointer))
          (alter to-pool #(assoc % pointer size)))))

(define (cuda-malloc-in-pool pointer size & pool)
 (JCuda/cudaMalloc pointer size)
 (cuda-add-to-pool! pointer size (if pool (first pool) *cuda-memory-pool*))
 pointer)

(define (cuda-free-pool & pool)
 (dorun
  (map #(jcuda.runtime.JCuda/cudaFree (first %))
       (dosync 
        (let ((m (if pool @(first pool) @*cuda-memory-pool*)))
         (ref-set (if pool (first pool) *cuda-memory-pool*) {})
         m)))))

;; Implementation

(define (cublas-allocate-zeroed ^Long size)
 (let ((ptr (jcuda.Pointer.)))
  (cuda-malloc-in-pool ptr (* size Sizeof/FLOAT))
  (JCuda/cudaMemset ptr 0 (* size Sizeof/FLOAT))
  ptr))

(define (cublas-allocate-from-float-array #^floats array)
 (let ((ptr (jcuda.Pointer.))
       (size (long (count array))))
  (cuda-malloc-in-pool ptr (* size Sizeof/FLOAT))
  ;; float-array fills with zeros by default
  (JCublas2/cublasSetVector size Sizeof/FLOAT (jcuda.Pointer/to array) 1 ptr 1)
  ptr))

(declare cuda-single?)
(declare to-matrix-cuda)
(declare cublas-clone-shape)
(declare cublas-unitialized-of-shape)
(declare cuda-launch-element-kernel2)
(declare cublas-lu-decomposition)

(define (with-cublas f)
 (let ((cublas-handle (cublasHandle.)))
  (JCublas2/cublasCreate cublas-handle)
  (let ((r (f cublas-handle)))
   (JCublas2/cublasDestroy cublas-handle)
   r)))

(deftype CudaSingle [^jcuda.Pointer pointer rows columns ld vec?]
 Object
 (toString [m] (str `(m/coerce :cuda-single ~(m/coerce :persistent-vector m))))
 ;; ld is in columns, cublas prefers column-major operations but we're
 ;; going to stick to row-major
 mp/PImplementation
 (implementation-key [m] :cuda-single)
 (meta-info [m] { :doc "JCuda" })
 (construct-matrix [m data]
   (if (= (type data) CudaSingle)
    (mp/clone data)
    (condp = (mp/dimensionality data)
     0 (float (mp/get-0d data))
     1 (CudaSingle. (cublas-allocate-from-float-array (float-array (mp/to-double-array data)))
                     (mp/dimension-count data 0)
                     1
                     1
                     true)
     2 (CudaSingle. (cublas-allocate-from-float-array (float-array (mp/to-double-array data)))
                     (mp/dimension-count data 0)
                     (mp/dimension-count data 1)
                     (mp/dimension-count data 1)
                     false)
     (throw (Exception. "Only supports vectors and 2D matrices")))))
 (new-vector [m length]
   (CudaSingle. (cublas-allocate-zeroed length) length 1 1 true))
 (new-matrix [m rows columns]
   (CudaSingle. (cublas-allocate-zeroed (* rows columns)) rows columns columns false))
 (new-matrix-nd [m shape]
   (case (int (count shape))
    1 (CudaSingle. (cublas-allocate-zeroed (first shape)) (int (first shape)) 1 1 true)
    2 (CudaSingle. (cublas-allocate-zeroed (* (first shape) (second shape)))
                    (int (first shape)) (int (second shape)) (int (second shape)) false)
    (throw (Exception. "Only supports scalars, 1D, and 2D matrices"))))
 (supports-dimensionality? [m dimensions] (or (= dimensions 1) (= dimensions 2)))
 mp/PDimensionInfo
 (dimensionality [m] (if vec? 1 2))
 (get-shape [m] (if vec? [rows] [rows columns]))
 (is-scalar? [m] false)
 (is-vector? [m] vec?)
 (dimension-count [m dimension-number]
   (condp = dimension-number
    0 rows
    1 (if vec? (error "Vectors are 1D") columns)
    (error "Can only represent 1D and 2D matrices")))
 mp/PIndexedAccess
 (get-1d [m row]
   ;; This would be ok for matrices if it handled submatrices
   (unless vec? (error "Can't use a 1D index on a matrix"))
   (let ((array (float-array [0])))
    (JCublas2/cublasGetVector 1 Sizeof/FLOAT
                              (pointer+ pointer (* row Sizeof/FLOAT ld)) 1
                              (jcuda.Pointer/to array) 1)
    (nth array 0)))
 (get-2d [m row column]
   (when vec? (error "Can't use a 2D on a vector"))
   (let ((array (float-array [0])))
    (JCublas2/cublasGetVector 1 Sizeof/FLOAT
                              (pointer+ pointer (* (+ (* row ld) column) Sizeof/FLOAT)) 1
                              (jcuda.Pointer/to array) 1)
    (nth array 0)))
 (get-nd [m indexes]
   (case (int (count indexes))
    1 (mp/get-1d m (first indexes))
    2 (mp/get-2d m (first indexes) (second indexes))
    (error (str "Unsupported number of indices" (into [] indexes)))))
 mp/PIndexedSetting
 (set-1d [m row v] (let ((m' (mp/clone m)))
                    (mp/set-1d! m' row v)
                    m'))
 (set-2d [m row column v] (let ((m' (mp/clone m)))
                           (mp/set-2d! m' row column v)
                           m'))
 (set-nd [m indexes v] (case (int (count indexes))
                        1 (mp/set-1d m (first indexes) v)
                        2 (mp/set-2d m (first indexes) (second indexes) v)
                        (error "Unsupported number of indices")))
 (is-mutable? [m] true)
 mp/PIndexedSettingMutable
 (set-1d! [m row v]
   ;; This would be ok for matrices if it handled submatrices
   (unless vec? (error "Can't 1D set a matrix"))
   (JCublas2/cublasSetVector 1 Sizeof/FLOAT
                             (jcuda.Pointer/to (float-array [v])) 1
                             (pointer+ pointer (* row Sizeof/FLOAT ld)) 1))
 (set-2d! [m row column v]
   (when vec? (error "Can't 2D set a vector"))
   (JCublas2/cublasSetVector 1 Sizeof/FLOAT
                             (jcuda.Pointer/to (float-array [v])) 1
                             (pointer+ pointer (* (+ (* row ld) column) Sizeof/FLOAT)) 1))
 (set-nd! [m indexes v]
   (case (int (count indexes))
    1 (mp/set-1d! m (first indexes) v)
    2 (mp/set-2d! m (first indexes) (second indexes) v)
    (error "Unsupported number of indices")))
 mp/PMatrixCloning
 (clone [m]
   (let ((ptr (jcuda.Pointer.))
         (size (* rows columns)))
    (cuda-malloc-in-pool ptr (* size Sizeof/FLOAT))
    (if (= columns ld)
     (JCuda/cudaMemcpy ptr pointer (* columns rows Sizeof/FLOAT) cudaMemcpyKind/cudaMemcpyDeviceToDevice)
     (JCuda/cudaMemcpy2D ptr (* columns Sizeof/FLOAT)
                         pointer (* ld Sizeof/FLOAT)
                         (* columns Sizeof/FLOAT) (* rows Sizeof/FLOAT)
                         cudaMemcpyKind/cudaMemcpyDeviceToDevice))
    (CudaSingle. ptr rows columns columns vec?)))
 ;; End of mandatory protocols
 mp/PTypeInfo
 (element-type [m] Float/TYPE)
 ;; (defprotocol PArrayMetrics
 ;;  "Option protocol for quick determination of array matrics"
 ;;  (nonzero-count [m]))
 ;; (defprotocol PValidateShape
 ;;   "Optional protocol to validate the shape of a matrix. If the matrix has an incorrect shape, should 
 ;;    throw an error. Otherwise it should return the correct shape."
 ;;   (validate-shape [m])) 
 mp/PRowColMatrix
 (column-matrix [m data]
   (let ((data ^CudaSingle (to-matrix-cuda data)))
    (unless (.vec? data) (error "Must be a 1D vector"))
    (let ((c ^CudaSingle (m/clone data)))
     (CudaSingle. (.pointer c) (.rows c) (.columns c) (.columns c) false))))
 (row-matrix [m data]
   (let ((data ^CudaSingle (to-matrix-cuda data)))
    (unless (.vec? data) (error "Must be a 1D vector"))
    (let ((c ^CudaSingle (m/clone data)))
     (CudaSingle. (.pointer c) (.columns c) (.rows c) (.rows c) false))))
 mp/PMutableMatrixConstruction
 (mutable-matrix [m]
   ;; FIXME does this have to make a copy if the original matrix is already mutable?
   (mp/clone m))
 mp/PSpecialisedConstructors
 ;; TODO Not great implementations, but much better than the defaults
 (identity-matrix [m dims]
   (let ((r (m/new-matrix m dims dims)))
    (dotimes [i dims] (mp/set-2d! r i i 1))
    r))
 (diagonal-matrix [m diagonal-values]
   (let ((r (m/new-matrix m (count diagonal-values) (count diagonal-values))))
    (dotimes [i (count diagonal-values)] (mp/set-2d! r i i (nth diagonal-values i)))
    r))
 mp/PReshaping
 (reshape
   [^CudaSingle m shape]
   (if (> (reduce * shape) (* (.rows m) (.columns m))) (error "Not enough elements"))
   (condp = (length shape)
    1 (CudaSingle. (.pointer m) (first shape) 1 1 true)
    2 (CudaSingle. (.pointer m) (first shape) (second shape) (second shape) false)
    (error "Can only do vectors or matrices")))
 mp/PMatrixMultiply
 (matrix-multiply [^CudaSingle a b]
   (let ((a ^CudaSingle (to-matrix-cuda a))
         (b ^CudaSingle (to-matrix-cuda b))
         (c ^CudaSingle (if (.vec? b)
                          (m/new-vector a (.rows a))
                          (m/new-matrix a (.rows a) (.columns b))))
         ;; FIXME Why do I need to specify this twice?
         (c ^CudaSingle c))
    (unless (= (.columns a) (.rows b)) (error "Can't multiply because matrices are incompatible"))
    (JCublas/cublasSgemm \T (if (.vec? b) \T \n)
                         (.rows a)
                         (.columns b)
                         (.columns a)
                         1.0
                         (.pointer a) (.ld a)
                         (.pointer b) (.ld b)
                         0.0
                         (.pointer c) (if (.vec? b)
                                       (.rows c)
                                       (.ld c)))
    c))
 (element-multiply [^CudaSingle m a]
   (let ((result (cublas-clone-shape m)))
    (cuda-launch-element-kernel2 "mul" m (to-matrix-cuda a) result)
    result))
 mp/PMatrixSlices
 (get-row [^CudaSingle m i]
   (unless (and (>= i 0) (< i rows))  (error "Out of bounds" i))
   (if vec?
    (CudaSingle. (pointer+ pointer (* Sizeof/FLOAT i)) 1 1 1 true)
    (CudaSingle. (pointer+ pointer (* Sizeof/FLOAT i ld)) columns 1 1 true)))
 (get-column [^CudaSingle m i]
   (when (not vec?) (error "Can only slice a matrix"))
   (unless (and (>= i 0) (< i columns))  (error "Out of bounds" i))
   (CudaSingle. (pointer+ pointer (* Sizeof/FLOAT i)) rows 1 columns true))
 (get-major-slice [m i] (mp/get-slice m 0 i))
 (get-slice [m dimension i]
   (if vec?
    (do (unless (= dimension 0)
         (error "Vectors are 1 dimensional"))
        (mp/get-1d m i))
    (condp = dimension
     0 (mp/get-row m i)
     1 (mp/get-column m i)
     (error "Only vectors and matrices are supported" i))))
 mp/PSliceView
 (get-major-slice-view [m i] (mp/get-major-slice m i))
 mp/PSliceSeq
 (get-major-slice-seq [m]
   (map-n #(mp/get-major-slice-view m %) (mp/dimension-count m 0)))
 mp/PSliceViewSeq
 (get-major-slice-view-seq [m] (mp/get-major-slice-seq m))
 mp/PMatrixSubComponents
 (main-diagonal [m]
   ;; TODO Not a great implementation, but much better than the default
   (if (.vec? m)
    (m/clone m)
    (let ((r (m/new-vector m (min rows columns))))
     (dotimes [i (min rows columns)] (mp/set-1d! r i (mp/get-2d m i i)))
     r)))
 mp/PSubVector
 (subvector [m start length]
   (unless vec? (error "Not a vector"))
   (unless (and (>= start 0) (> length 0)
                (< start rows) (<= (+ start length) rows))
    (error "Out of bounds"))
   (CudaSingle. (pointer+ pointer (* Sizeof/FLOAT start)) length 1 1 true))
 mp/PVectorisable
 (to-vector [m]
   (if vec?
    m
    (if (= columns ld)
     (CudaSingle. pointer (* columns rows) 1 1 true)
     (error "BLAS operations are not compatible with views of submatrices"))))
 mp/PDoubleArrayOutput
 (to-double-array [m]
   ;; TODO This copies twice, but is still far more efficient (~200x) than the default
   (let ((size (reduce * (m/shape m)))
         (array (make-array Float/TYPE size)))
    (if (= columns ld)
     (JCuda/cudaMemcpy (jcuda.Pointer/to array) pointer (* columns rows Sizeof/FLOAT) cudaMemcpyKind/cudaMemcpyDeviceToHost)
     (JCuda/cudaMemcpy2D (jcuda.Pointer/to array) (* columns Sizeof/FLOAT)
                         pointer (* ld Sizeof/FLOAT)
                         (* columns Sizeof/FLOAT) (* rows Sizeof/FLOAT)
                         cudaMemcpyKind/cudaMemcpyDeviceToHost))
    (double-array array)))
 (as-double-array [m] nil)
 mp/PObjectArrayOutput
 ;; Could save one copy here, but I don't see anyone using this anyway
 (to-object-array [m] (object-array (mp/to-double-array m)))
 (as-object-array [m] nil)
 mp/PTranspose
 (transpose [m]
   (if (= (mp/dimensionality m) 2)
    (let ((m' ^CudaSingle (cublas-clone-shape ^CudaSingle m :transpose? true)))
     (with-cublas
      (lambda (cublas-handle)
       (JCublas2/cublasSgeam cublas-handle
                             cublasOperation/CUBLAS_OP_T
                             cublasOperation/CUBLAS_OP_N
                             (.rows m)
                             (.columns m)
                             (jcuda.Pointer/to (float-array [1]))
                             (.pointer m)
                             (.ld m)
                             (jcuda.Pointer/to (float-array [0]))
                             (.pointer m)
                             (.ld m)
                             (.pointer m')
                             (.ld m'))))
     m')
    m))
 mp/PNumerical
 (numerical? [m] true)
 mp/PVectorOps
 (vector-dot [^CudaSingle m a]
   (let ((a ^CudaSingle (m/coerce :cuda-single a)))
    (unless (and (.vec? m) (.vec? a) (= (.rows m) (.rows a)))
     (error "Arguments must be vectors of the same size"))
    (JCublas/cublasSdot
     (.rows m)
     (.pointer m) (.ld m)
     (.pointer a) (.ld a))))
 (length [a]
   (unless (.vec? a) (error "Can only compute the lenght of vectors"))
   (JCublas/cublasSnrm2 (.rows a) (.pointer a) (.ld a)))
 (length-squared [a]
   ;; TODO This could potentially be more efficient but cublas kernels
   ;;      are much better than my own
   (sqr (mp/length a)))
 (normalise [a] (mp/scale a (/ 1 (mp/length a))))
 mp/PMutableVectorOps
 (normalise! [a] (mp/scale! a (/ 1 (mp/length a))))
 mp/PVectorDistance
 (distance [a b] (mp/length (mp/matrix-sub a b)))
 mp/PMatrixScaling
 (scale [m a]
   (unless (number? a) (error "Can only scale by a number" m a))
   (if (= ld columns)
    (let ((r ^CudaSingle (mp/new-matrix-nd m (mp/get-shape m))))
     ;; FIXME Could use cublasSscal instead
     (JCublas/cublasSaxpy (* rows columns) a pointer 1 (.pointer r) 1)
     r)
    (error "scale is not (yet) supported for matrix views")))
 (pre-scale [m a] (mp/scale m a))
 mp/PMatrixMutableScaling
 (scale! [m a]
   (unless (number? a) (error "Can only scale by a number" m a))
   (unless (= ld columns) (error "scale is not (yet) supported for matrix views"))
     ;; FIXME Could use cublasSscal instead
   (JCublas/cublasSscal (* rows columns) a pointer 1)
   m)
 (pre-scale! [m a] (mp/scale! m a))
 mp/PMatrixAdd
 (matrix-add [m a]
   (let ((result (cublas-clone-shape m)))
    (cuda-launch-element-kernel2 "add" m (to-matrix-cuda a) result)
    result))
 (matrix-sub [m a]
   (let ((result (cublas-clone-shape m)))
    (cuda-launch-element-kernel2 "sub" m (to-matrix-cuda a) result)
    result))
 mp/PMatrixAddMutable
 (matrix-add! [m a]
   (cuda-launch-element-kernel2 "add" m (to-matrix-cuda a) m)
   m)
 (matrix-sub! [m a]
   (cuda-launch-element-kernel2 "sub" m (to-matrix-cuda a) m)
   m)
 mp/PSummable
 (element-sum [m]
   (let ((tmp ^CudaSingle (m/coerce :cuda-single [1])))
    (if vec?
     ;; This is a hack! inc == 0 is ok here!
     (JCublas/cublasSdot rows pointer ld (.pointer tmp) 0)
     (if (= ld columns)
      ;; This is a hack! inc == 0 is ok here!
      (JCublas/cublasSdot (* rows columns) pointer 1 (.pointer tmp) 0)
      (error "element-sum is not (yet) supported for vector or matrix views"))))))
 
(defmulti to-kernel-pointer type)
;; The annoying repetition of jcuda.Pointer/to is needed because it
;; avoids runtime reflection
(defmethod to-kernel-pointer Float [n] (jcuda.Pointer/to (float-array [n])))
(defmethod to-kernel-pointer Double [n]
 ;; Note! All of the code here is single-precision, so Clojure doubles
 ;; become float-arrays not double arrays
 (jcuda.Pointer/to (float-array [n])))
(defmethod to-kernel-pointer Long [n] (jcuda.Pointer/to (long-array [n])))
(defmethod to-kernel-pointer Integer [n] (jcuda.Pointer/to (int-array [n])))
(defmethod to-kernel-pointer CudaSingle [^CudaSingle j]
 ;; FIXME There is reflection here, but I don't know how to avoid it
 (jcuda.Pointer/to (into-array jcuda.NativePointerObject [(.pointer j)])))

(defnk cuda-launch-kernel
 [module function parameters
  :gridX 1 :gridY 1 :gridZ 1 :blockX 1 :blockY 1 :blockZ 1
  :sharedMemory 0 :stream nil]
 (let ((f (CUfunction.)))
  (JCudaDriver/cuModuleGetFunction f module function)
  (JCudaDriver/cuLaunchKernel
   f
   gridX gridY gridZ
   blockX blockY blockZ
   sharedMemory stream
   ;; FIXME This is reflection here, but I don't know how to avoid it
   (jcuda.Pointer/to
    (into-array jcuda.NativePointerObject
                (map (lambda (p) (to-kernel-pointer p))
                     parameters)))
   nil)))

(defn ^CudaSingle cuda-launch-element-kernel2
 [function-name ^CudaSingle m ^CudaSingle a ^CudaSingle result]
 (JCudaDriver/cuCtxSetCurrent (:context *default-cuda-context*))
 (let ((tpb (:maxThreadsPerBlock *cuda-default-device-properties*))
       (ws (:warpSize *cuda-default-device-properties*))
       (kernel
        (lambda (function-postfix args R C)
         (cuda-launch-kernel (:kernels-module *default-cuda-context*)
                             (str "e" function-name "_" function-postfix)
                             args
                             :gridX (Math/ceil (/ R tpb))
                             :blockX (min (* (Math/ceil (/ R ws)) ws) tpb)
                             :gridY (Math/ceil (/ C tpb))
                             :blockY (min (if (= C 1) 1 (* (Math/ceil (/ C ws)) ws)) tpb)))))
  (condp = [(m/dimensionality m) (m/dimensionality a)]
   [1 0] (kernel "vsf" [(.rows m) m 1 a result 1] (.rows m) 1)
   [1 1] (do (unless (= (m/dimensionality m) (m/dimensionality a))
              (error "Incompatible dimensionality"))
             (kernel "vvf" [(.rows m) m 1 a 1 result 1] (.rows m) 1))
   [1 2] (do (unless (= (.rows m) (.columns a))
              (error "Incompatible dimensionality"))
             (kernel "vmf" [(.rows a) (.columns a) m 1 a (.ld a) result (.ld result)]
                     (.columns a) (.rows a)))
   [2 0] (kernel "msf" [(.rows m) (.columns m) m (.ld m) a result (.ld result)]
                 (.columns m) (.rows m))
   [2 1] (do (unless (= (.rows m) (.columns a))
              (error "Incompatible dimensionality"))
             (kernel "mvf" [(.rows m) (.columns m) m (.ld m) a 1 result (.ld result)]
                     (.columns m) (.rows m)))
   [2 2] (do (unless (= (m/dimensionality m) (m/dimensionality a))
              (error "Incompatible dimensionality"))
             (kernel "mmf" [(.rows a) (.columns a) m (.ld m) a (.ld a) result (.ld result)]
                     (.columns a) (.rows a))))
  (JCudaDriver/cuCtxSynchronize)
  result))

(defnk cublas-clone-shape [^CudaSingle m :transpose? false]
 (let ((ptr (jcuda.Pointer.))
       (size (* (.rows m) (.columns m))))
  (cuda-malloc-in-pool ptr (* size Sizeof/FLOAT))
  (if transpose?
   (CudaSingle. ptr (.columns m) (.rows m) (.rows m) (.vec? m))
   (CudaSingle. ptr (.rows m) (.columns m) (.columns m) (.vec? m)))))

(define (cublas-uninitialized-of-shape rows columns vec?)
 (let ((ptr (jcuda.Pointer.))
       (size (* rows columns)))
  (cuda-malloc-in-pool ptr (* size Sizeof/FLOAT))
  (CudaSingle. ptr rows columns columns vec?)))

(defn- ^CudaSingle to-matrix-cuda [m]
 (if (cuda-single? m)
  m
  (mp/construct-matrix (imp/get-canonical-object :cuda-single) m)))

(define (cuda-single? data) (= (type data) CudaSingle))

(binding
  [*cuda-memory-pool* *cuda-permanent-pool*]
 (imp/register-implementation (CudaSingle. (cublas-allocate-zeroed 1) 1 1 1 true)))
