(ns nets.core
  (:require [nets.backpropogation :as backprop]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns])
  (:gen-class))

(defn perror
  "Calculates the percent error given an ERROR and a TARGET value.
  Currently this only works for the l2-metric."
  [output target training-profile]
  (let [zero-vec (take (count target) (repeat 0))
        cost-fn (error-fns/get-cost-fn :mean-squared)]
    (/ (cost-fn output target)
       (cost-fn target zero-vec))))

(defn- float-stringer
  "Converts the vector of floats VEC into a vector of strings of N places after
  the decimal point. EG:
  (vec-> str [0.123 0.123] 2) ==> [\"0.12\" \"0.12\"]"
  [n x]
  (let [fstring (str "%." (format "%d" n) "f")]
    (if (vector? x)
      (mapv #(format fstring %) (mapv float x))
      (format fstring (float x)))))

(defn sample-printer
  [inputs outputs targets errors perrors]
  (let [float-formatter (partial float-stringer 5)
        v (mapv #(mapv float-formatter %) [inputs outputs targets errors perrors])]
    (clojure.pprint/print-table
     (apply
      (partial mapv #(zipmap [:input :output :target :error :perror]
                             (vector %1 %2 %3 %4 %5)))
      v))))

(defn sample-output
  "Produces the output and the error for N iterations of NET under profile
  TRAINING-PROFILE"
  [net training-profile sample-count]
  (let [inputs (take sample-count (repeatedly (:input-fn training-profile)))
        outputs (mapv #(backprop/net-eval net %) inputs)
        targets (mapv (:output-fn training-profile) inputs)
        errors (mapv (error-fns/get-cost-fn (:cost-fn training-profile))
                     targets outputs)
        perrors (mapv #(perror %1 %2 training-profile) outputs targets)]
    (sample-printer inputs outputs targets errors perrors)))

(defn train-for
  [net training-profile iterations]
  (let [input ((:input-fn training-profile))
        output ((:output-fn training-profile) input)
        next-net (backprop/train net input output
                                 (:lrate training-profile)
                                 (error-fns/get-cost-grad
                                  (:cost-fn training-profile)))]
    (if (>= 1 iterations)
      ; hardcoded 20
      (sample-output next-net training-profile 20)
      (recur next-net training-profile (dec iterations)))))
