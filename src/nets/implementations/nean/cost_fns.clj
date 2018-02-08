(ns nets.implementations.nean.cost-fns
  (:require [uncomplicate.neanderthal.vect-math :as vmath])
  (:use uncomplicate.fluokitten.core
        uncomplicate.neanderthal.native
        uncomplicate.neanderthal.core))

(defn- mean-squared
  "Returns the mean squared error of IDEAL and ACTUAL."
  [actual-v targ-v]
  (* 0.5
     (sum (vmath/pow
           (axpy -1.0 targ-v actual-v)
           2))))

(defn- mean-squared-grad
  "The vector derivative of the mean-squared-error function. Component
  i of the return vector is the derivative of the mean-squared-error
  function with respect to the i_th output neuron."
  [actual ideal]
  (axpy -1.0 ideal actual))

(defn- v- ^double
  [^double x]
  (- 1.0 x))

(defn- one-minus-v
  [v]
  (fmap v- v))

(defn- cross-entropy
  "The cross entropy cost function is defined as
   - (sum [e_i * ln (a_i) + (1 - e_i) ln (1 - a_j).
  where E = [e_0 ... e_j] is the expected value and
  A = [a_0 ... a_j] is the observed value."
  [actual ideal]
  (let [xs (vmath/mul ideal (vmath/log actual))
        ys (vmath/mul (one-minus-v ideal)
                      (vmath/log (one-minus-v actual)))]
    (- (+ (sum xs) (sum ys)))))

(defn cent-helper ^double
  [^double x]
  (if (< x 0.0001)
    0.0001
    (* x (- 1.0 x))))

(defn- cross-entropy-grad
  "Given expected output E and observed output A
  the gradient of the cross entropy function is V
  where v_j = (a_j - e_j)/(1 - a_j)a_j."
  [actual ideal]
  (vmath/div
        (axpy -1.0 ideal actual)
        (fmap cent-helper actual)))

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
