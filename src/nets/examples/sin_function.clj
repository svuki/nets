(ns nets.examples.sin-function
  (:require [nets.net :as net]
            [nets.sgd-handler :as sgd]
            [nets.implementations.nean.interface :as nean]
            [nets.utils.printers :as printers]))

(defn sin-in
  "Returns a random input in [0, 2pi)."
  []
  (rand (* 2 Math/PI)))

(defn sin-out
  "Normalized sin function to [0,1]. "
  [x]
  (* 0.5 (inc (Math/sin x))))

(def training-points
  (reify clojure.lang.ISeq
    (first  [_] ((fn [] (let [inp (sin-in)]
                          (list [inp] [(sin-out inp)])))))
    (more   [_] training-points)
    (empty  [_] (list))
    (cons   [& rest] training-points)
    (equiv  [& rest] false)
    (next   [_] training-points)
    (seq    [_] training-points)))

(def sin-tprofile
  {:lrate 0.2
   :cost-fn :mean-squared
   :training-data training-points})

(defn sin-example
  [iterations]
  (let [test-net (net/new-net 1 [40 :leaky-relu] [1 :sigmoid])]
    (sgd/sgd-handler test-net sin-tprofile iterations nean/interface)))

(defn sample
  [n net tpoints]
  (let [points (take n tpoints)
        inps (map first points)
        targs (map second points)
        outs (map #((:net-eval nean/interface) net %) inps)
        errors (map #(- (first %1) (first %2))
                     outs targs)]
    (printers/sidewise
     ["Input" "Output" "Target" "Errors"]
     [inps
      outs
      targs
      errors])))
