(ns nets.examples.always-one
  (:require [nets.net :as net]
            [nets.sgd-handler :as sgd]
            [nets.implementations.nean.interface :as nean]
            [nets.utils.printers :as printers]))

(def training-points
  (reify clojure.lang.ISeq
    (first  [_] (list [(rand 1.0) (rand 1.0)] [1.0]))
    (more   [_] training-points)
    (empty  [_] (list))
    (cons   [& rest] training-points)
    (equiv  [& rest] false)
    (next   [_] training-points)
    (seq    [_] training-points)))
    
(def always-one-tprofile
  {:lrate 0.2
   :cost-fn :mean-squared
   :training-data training-points})

(defn always-one
  "In this exmaple, the net is trained to always output [1]."
  [times-to-run]
  (let [test-net (net/new-net 2 [10 :relu] [1 :sigmoid])]
    (sgd/sgd-handler test-net always-one-tprofile times-to-run nean/interface)))

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
                       
        
  
