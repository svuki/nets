;;; This file contains various activation functions and their derivatives.
;;; Other modules interact with this module via the functions get-fn and get-deriv
;;; which search the dynamic variable ACT-FNS for functiions associated with a given
;;; name. To add another activation function, use ADD-FUNCTIONS.
(ns nets.activation-functions
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix])
  (:gen-class))

(defn vectorize
  "Takes a function and returns its vectorized form."
  [func]
  (fn [v] (mapv func v)))

(defn no-NaN?
  "Ensures x is not NaN"
  [x]
  (not (Double/isNaN x)))
(defn no-NaNs?
  "Ensures no element in v is NaN"
  [v]
  (not-any? #(Double/isNaN %) v))

(defn sigmoid [x] (/ 1.0 (+ 1 (math/exp (- x)))))
(defn sigmoid-deriv [x] (* (sigmoid x) (- 1 (sigmoid x))))

(defn tanh [x] (- (* 2 (sigmoid (* 2 x))) 1))
(defn- tanh-deriv [x] (- 1 (math/pow (sigmoid x) 2)))

(defn- relu [x] (max 0 x))
(defn- relu-deriv [x] (if (> x 0) 1.0 0.0))

(defn- leaky-relu [x] (max 0 x))
(defn- leaky-relu-deriv
  "Similar to rectified linear units, but the derivative is
   a small number (implemented as (* 0.01 x)) for X <= 0."
  [x]
  (cond (> x 0) 1.0
        (<= x 0) (* 0.01 x)))

(defn- softplus
  [x]
  (Math/log (+ 1.0 (Math/exp x))))
(defn- softplus-grad
  [x]
  (/ 1.0
    (+ 1 (Math/log (- x)))))

(defn- softmax
  "Implements the softmax function. Note that this is a vector valued
   function. Given argument v of dimension K, the i_th component of the
   return vector is (e^(v_i) / (sum (j = 0 to K) e^(v_j))."
  [v]
  {:post [(no-NaNs? %)]}
  ; to avoid overflows we subtract the largest x from each of the values
  ; before exponentiating. Note that this does not change the value of
  ; the output
  (let [x (apply max v)
        exps (mapv #(math/exp (- % x)) v)
        sum (apply + exps)]
    (mapv #(/ % sum) exps)))

;;; TODO: when associated with certain loss functions, the computation of
;;; the soft-max "gradient" can be simplified. Detect when such
;;; such functions are used in unison and do the cheaper computatoins
(defn- kdelta
  "The kronecker delta. Returns 1.0 if i == j and 0.0 otherwise."
  [i j]
  (if (= i j) 1.0 0.0))

(defn- make-matrix
  "Takes integer M, integer N, and a function F of two arguments.
  Produces a matrix of M rows and N columns such that the (i,j) entry
  is (F i j)"
  [m n f]
  (let [mat-indices (map (fn [i] (map (fn [j] [i j]) (range n))) (range m))]
    (mapv #(mapv (fn [[i j]] (f i j)) %) mat-indices)))

;; TODO: check that this should in fact be the tranpose
(defn- softmax-jacobian
  "Given a vector v, returns the tranpose of the jacobian matrix
  of the softmax function evaluated at v."
  [v]
  ; To keep with the convention that we multiply by row vectors on the left
  ; we transpose the jacboian so that the first column gives the partial
  ; derivatives of the first component function of softmax with respect
  ; to the the inputs v_0 .. v_n.
  (matrix/transpose
   (make-matrix (count v) (count v)
                (fn [i j] (* (nth v i)
                             (- (kdelta i j)
                                (nth v j)))))))

(def ^:dynamic act-fns (transient {}))
(assoc! act-fns :sigmoid (mapv vectorize [sigmoid sigmoid-deriv]))
(assoc! act-fns :tanh (mapv vectorize [tanh tanh-deriv]))
(assoc! act-fns :relu (mapv vectorize [relu relu-deriv]))
(assoc! act-fns :softmax [softmax softmax-jacobian])
(assoc! act-fns :softplus (mapv vectorize [softplus softplus-grad]))

(defn get-fn
  "Returns the function associated with NAME. Returns nil if no function is found."
  [name]
  (first (get act-fns name)))

; TODO: function should be renamed to get-gradient or get-grad
(defn get-deriv
  "Returns the derivativ eof the function associated with NAME. Returns nil if no function is found."
  [name]
  (second (get act-fns name)))

(defn add-function
  "Enables the function FN and its derivative DERIVA to be found using get-fn and get-deriv."
  [name fn deriv]
  (assoc! act-fns name [fn deriv]))
