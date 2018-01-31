;;; This file contains various activation functions and their derivatives.
;;; Other modules interact with this module via the functions get-fn and get-deriv
;;; which search the dynamic variable ACT-FNS for functiions associated with a given
;;; name. To add another activation function, use ADD-FUNCTIONS.
(ns nets.activation-functions
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [nets.utilis :as utils]
            [nets.matrix-utils :as m-utils])
  (:gen-class))

(defn- vectorize
  "Takes a function and returns its vectorized form."
  [func]
  (fn [v] (matrix/emap func v)))

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
  {:post [(utils/no-NaNs? %)]}
  ; to avoid overflows we subtract the largest x from each of the values
  ; before exponentiating. Note that this does not change the value of
  ; the output
  (let [x (apply max v)
        exps (mapv #(math/exp (- % x)) v)
        sum (apply + exps)]
    (mapv #(/ % sum) exps)))

;; TODO: sloppy. In what cases would we use softmax without cross entropy? Would we ever use
;; it in a hidden layer? Right now cross-entropy/softmax on output is handled by a seperate
;; routine so the softmax-jacobian is never calculated
(defn- softmax-jacobian
  "Given a vector v, returns the tranpose of the jacobian matrix
  of the softmax function evaluated at v."
  [v]
  ; To keep with the convention that we multiply by row vectors on the left
  ; we transpose the jacboian so that the first column gives the partial
  ; derivatives of the first component function of softmax with respect
  ; to the the inputs v_0 .. v_n.
  (matrix/transpose
   (m-utils/make-matrix (count v) (count v)
                        (fn [i j] (* (nth v i)
                                     (- (m-utils/kdelta i j)
                                (nth v j)))))))

(def act-fns
  {:sigmoid (mapv vectorize [sigmoid sigmoid-deriv])
   :tanh (mapv vectorize [tanh tanh-deriv])
   :relu (mapv vectorize [relu relu-deriv])
   :leaky-relu (mapv vectorize [leaky-relu leaky-relu-deriv])
   :softmax [softmax softmax-jacobian]
   :softplus (mapv vectorize [softplus softplus-grad])})

(defn get-fn
  "Returns the function associated with NAME. Returns nil if no function is found."
  [name]
  (first (get act-fns name)))

; TODO: function should be renamed to get-gradient or get-grad
(defn get-deriv
  "Returns the derivativ eof the function associated with NAME. Returns nil if no function is found."
  [name]
  (second (get act-fns name)))
