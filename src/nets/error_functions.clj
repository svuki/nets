(ns nets.error-functions
  (:require [clojure.algo.generic.math-functions :as math])
  (:gen-class))

;;; This namespace provides various cost functions and their gradients
;;; By convention, the first argument will be the actual value and the
;;; second argument the target value

(defn- mean-squared
  "Returns the mean squared error of IDEAL and ACTUAL."
  [actual ideal]
  (* 0.5
     (reduce +
             (mapv #(Math/pow (- %1 %2) 2)
                   actual
                   ideal))))

(defn- mean-squared-grad
  "The vector derivative of the mean-squared-error function. Component
  i of the return vector is the derivative of the mean-squared-error
  function with respect to the i_th output neuron."
  [actual ideal]
  (mapv - actual ideal))

(def cost-fns
  {:mean-squared [mean-squared mean-squared-grad]})

(defn get-cost-fn
  [kw]
  (let [ret (get cost-fns kw nil)]
    (if ret
      (first ret)
      (throw (Throwable. "No such cost function is registered.")))))
(defn get-cost-grad
  [kw]
  (let [ret (get cost-fns kw nil)]
    (if ret
      (second ret)
      (throw (Throwable. "No such cost gradient function is registed.")))))

(defn showall-cost-fns []
  (keys cost-fns))
