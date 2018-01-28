;;; This file contains various activation functions and their derivatives.
;;; Other modules interact with this module via the functions get-fn and get-deriv
;;; which search the dynamic variable ACT-FNS for functiions associated with a given
;;; name. To add another activation function, use ADD-FUNCTIONS.
(ns nets.activation-functions
  (:require [clojure.algo.generic.math-functions :as math])
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

;;; TODO: softmax has a tendency to return NaN. Implement a normalization to keep the return value
;;; reasonable
(defn- softmax
  "Implements the softmax function. Note that this is a vector valued
   function. Given argument v of dimension K, the i_th component of the
   return vector is (e^(v_i) / (sum (j = 0 to K) e^(v_j))."
  [v]
  {:post [(no-NaNs? %)]}
  (let [exps (mapv math/exp v)
        sum (apply + exps)]
    (mapv #(/ % sum)
          exps)))

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

(defn- softmax-jacobian
  "Given a vector v, returns the jacobian matrix of the softmax function
  evaluated at v."
  [v]
  (make-matrix (count v) (count v)
               (fn [i j] (* (nth v i)
                            (- (kdelta i j)
                               (nth v j))))))

(def ^:dynamic act-fns (transient {}))
(assoc! act-fns :sigmoid (mapv vectorize [sigmoid sigmoid-deriv]))
(assoc! act-fns :tanh (mapv vectorize [tanh tanh-deriv]))
(assoc! act-fns :relu (mapv vectorize [relu relu-deriv]))


; TODO: what if result is nil?
(defn get-fn
  "Returns the function associated with NAME. Returns nil if no function is found."
  [name]
  (first (get act-fns name)))

; TODO: what if result is nil?
(defn get-deriv
  "Returns the derivativ eof the function associated with NAME. Returns nil if no function is found."
  [name]
  (second (get act-fns name)))

(defn add-function
  "Enables the function FN and its derivative DERIVA to be found using get-fn and get-deriv."
  [name fn deriv]
  (assoc! act-fns name [fn deriv]))
