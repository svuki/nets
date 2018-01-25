(ns nets.error-functions
  (:require [clojure.algo.generic.math-functions :as math])
  (:gen-class))

;;; This namespace provides various error functions and their derivatives

(defn l2 ;vectorized
  [ideal actual]
  (* 0.5 (reduce + (mapv #(math/pow (- %1 %2) 2) ideal actual))))
(defn l2-deriv ;vectorized
  [ideal actual]
  (- ideal actual))
