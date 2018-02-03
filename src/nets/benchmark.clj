(ns nets.benchmark
  (require [nets.backpropogation :as b]
           [nets.net :as net]
           [clojure.core.matrix :as mat]
           [clojure.core.matrix.implementations :as mat-imp])
  (use nets.utils)
  (use nets.matrix-utils))



(defn input-fn []
  (mat/array (repeatedly 800 #(rand 1))))
(defn output-fn [& rest]
  (mat/array [0 0 0 0 1 0 0 0 0 0]))

(def bench-tprofile
  {:lrate 0.2
   :cost-fn :cross-entropy
   :input-fn input-fn
   :output-fn output-fn})

(defn validate [net]
  (let [check? #(= 4 (predict %))]
    (/ (reduce + (repeatedly 100 #(check? (b/net-eval net (input-fn)))))
       100.0)))

(mat-imp/set-current-implementation :clatrix)
(defn run-bench []
  (print "Current implementation " (mat/current-implementation) "\n")
  (flush)
  (let [b-net (net/new-net 800 [800 :sigmoid] [10 :softmax])]
    (time
     (loop [n b-net
            count 100]
       (if (zero? count)
         n
         (recur (b/sgd n bench-tprofile) (dec count)))))))
