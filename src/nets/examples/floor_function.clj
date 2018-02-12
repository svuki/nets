(ns nets.examples.floor-function
  (:require [nets.net :as net]
            [nets.sgd-handler :as sgd]
            [nets.implementations.nean.interface :as nean]
            [nets.utils.printers :as printers]
            [nets.utils.matrix-utils :as mutils]))

(def training-points
  (reify clojure.lang.ISeq
    (first  [_] ((fn [] (let [inp (rand 5)]
                          (list [(/ inp 5.0)] (mutils/one-hot (Math/floor inp) 5))))))
    (more   [_] training-points)
    (empty  [_] (list))
    (cons   [& rest] training-points)
    (equiv  [& rest] false)
    (next   [_] training-points)
    (seq    [_] training-points)))

(def floor-tprofile
  {:lrate 0.2
   :cost-fn :cross-entropy
   :training-data training-points})

(defn floor-example
  [iterations]
  (let [test-net (net/new-net 1 [30 :relu] [5 :softmax])]
    (sgd/sgd-handler test-net floor-tprofile iterations nean/interface)))

(defn sample
  [n net tpoints]
  (let [points (take n tpoints)
        inps (map first points)
        targs (map second points)
        outs (map #((:net-eval nean/interface) net %) inps)]
    (printers/sidewise
     ["Input" "Output" "Target"]
     [inps
      outs
      targs])))
