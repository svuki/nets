(ns nets.examples
  (:require [nets.core :as core]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns])
  (:gen-class))

(defn sample-input-fn
  []
  [(rand 1) (rand 1)])

(defn sample-output-fn
  [_]
  [1])

(defn always-one
  "In this exmaple, the net is trained to always output [1]."
  [learning-rate times-to-run]
  (let [test-net (net/new-net 2 [2 :sigmoid] [1 :sigmoid])]
    (core/multi-set-train-verbose
     test-net sample-input-fn sample-output-fn
     error-fns/l2 error-fns/l2-deriv learning-rate times-to-run)))

(defn reverse-fn
  [[x y]] [y x])

(defn always-reverse
  "In this example the net is trained to learn the function
  [x y] --> [y x]."
  [learning-rate times-to-run]
  (let [test-net (net/new-net 2 [2 :tanh] [2 :tanh])]
    (core/multi-set-train-verbose
     test-net sample-input-fn reverse-fn
     error-fns/l2 error-fns/l2-deriv learning-rate times-to-run)))
