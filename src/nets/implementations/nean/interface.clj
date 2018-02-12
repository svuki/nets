(ns nets.implementations.nean.interface
  (:require [nets.implementations.nean.sgd :as sgd]
            [nets.implementations.nean.activation-functions :as afs]
            [nets.implementations.nean.cost-fns :as cfs]
            [nets.net :as net])
  (:use uncomplicate.neanderthal.core
        uncomplicate.neanderthal.native
        nets.utils.utils))

; TODO: implement a reverse map lookup to determine the activation function keyword
; so that it doesn't need to be saved
(defn from-net
  "Returns a vector of four vectors:
  [matrices biases activation-fns deacitvation-fns]."
  [net-spec]
  (let [mats   (mapv #(ge native-double
                          (count %)
                          (count (first %))
                          (flatten %))
                     (net/matrices net-spec))
        biases (mapv dv (net/biases net-spec))
        afs    (mapv afs/get-fn (net/act-fns net-spec))
        dfs    (mapv afs/get-deriv (net/act-fns net-spec))]
    [mats biases afs dfs]))

(defn to-net
  "Returns a net description."
  [[ms bs] af-names]
  {:num-inputs (dim (row (first ms) 0))
   :matrices (mapv #(map seq (rows %)) ms)
   :biases (mapv seq bs)
   :act-fns af-names})

(defn from-data-pts
  "Converts pairs of sequences to pairs of Neanderthal double precision vectors."
  [training-data]
  (map-pairs dv training-data))

(defn run-sgd
  [net {lrate :lrate tdata :training-data cfun :cost-fn} iterations]
  {:pre [(net/description? net)
         (keyword? cfun)
         (not (neg? iterations))
         lrate tdata]}
  (binding
      [sgd/*smax-cent?* (if (and (= cfun :cross-entropy)
                                 (= (last (net/act-fns net)) :softmax))
                          true
                          false)]
    (to-net 
     (sgd/sgd (from-net net)
              (from-data-pts tdata)
              lrate
              (cfs/get-cost-grad cfun)
              iterations)
     (net/act-fns net))))

(defn net-eval
  [net input]
  (seq
   (if (net/description? net)
     (net-eval (from-net net) (dv input))
     (sgd/net-eval net (dv input)))))

(def interface
  {:from-net from-net 
   :run-sgd  run-sgd 
   :net-eval net-eval
   :to-vec   seq
   :from-vec dv})
