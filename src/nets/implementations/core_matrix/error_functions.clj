(ns nets.core-matrix.error-functions
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix])
  (:gen-class))

;;; This namespace provides various cost functions and their gradients
;;; By convention, the first argument to the cost functions and their gradients
;;; will be the OBSERVED value and the second argument will be the EXPECTED value.
;;; All cost functions should have type VEC -> VEC -> REAL while their gradients
;;; should have type VEC -> VEC -> VEC.

(defn- mean-squared
  "Returns the mean squared error of IDEAL and ACTUAL."
  [actual ideal]
  (* 0.5
     (reduce + (matrix/pow
                (matrix/sub actual ideal)
                2))))

(defn- mean-squared-grad
  "The vector derivative of the mean-squared-error function. Component
  i of the return vector is the derivative of the mean-squared-error
  function with respect to the i_th output neuron."
  [actual ideal]
  (matrix/sub actual ideal))

(defn- cross-entropy
  "The cross entropy cost function is defined as
   - (sum [e_i * ln (a_i) + (1 - e_i) ln (1 - a_j).
  where E = [e_0 ... e_j] is the expected value and
  A = [a_0 ... a_j] is the observed value."
  [actual ideal]
  (let [xs (matrix/emul ideal (matrix/emap (fn [x] (Math/log x)) actual))
        ys (matrix/emul (matrix/emap #(- 1.0 %) ideal)
                        (matrix/emap (fn [x] (Math/log (- 1.0 x))) actual))]
    (- (reduce + (into ys xs)))))

; Todo: rewrite in trems of core.matrix abstraction
; TODO: at times the gradient of the cross entropy function results in a divison
; by zero error. The ideal workaround would be to combine the softmax and
; cross-entropy calculation to prevent extreme values in the calculation. For now
; cross-entropy-grad will replace 0's in its denominator with small values
(defn- cross-entropy-grad
  "Given expected output E and observed output A
  the gradient of the cross entropy function is V
  where v_j = (a_j - e_j)/(1 - a_j)a_j."
  [actual ideal]
  (mapv #(/ %1 %2)
        (mapv - actual ideal)
        (mapv
         (fn [a] (if (< a 0.01)
                   0.01
                   (* a (- 1.0 a))))
           actual)))

(def cost-fns
  {:mean-squared [mean-squared mean-squared-grad]
   :cross-entropy [cross-entropy cross-entropy-grad]})

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
