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


(defn sigmoid [x] (/ 1.0 (+ 1 (math/exp (- x)))))
(defn sigmoid-deriv [x] (* (sigmoid x) (- 1 (sigmoid x))))

(defn tanh [x] (- (* 2 (sigmoid (* 2 x))) 1))
(defn- tanh-deriv [x] (- 1 (math/pow (sigmoid x) 2)))

(defn- relu [x] (max 0 x))
(defn- relu-deriv [x] (if (> x 0) 1.0 0.0))
;;; TODO: specify how relu-deriv is handled, provide options for alternative implementations


;;; TODO: unlike the other functions, softmax is a vector function, meaning its derivative cannot be implemented componentwise.
(defn- softmax [v] ; Note that this is already in vectorized form
  (let [exps (mapv math/exp v)
        sum (reduce + exps)]
    (mapv #(/ % sum) exps)))

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
