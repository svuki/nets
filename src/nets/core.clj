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

(defn- to-string
  "Converts floats into strings of N decimal length. Converts
  vectors of numbers into a single string for printing."
  [n x]
  (let [fstring (str "%." (format "%d" n) "f")]
    (if (vector? x)
      (apply str
             (interpose
              " "
              (mapv #(format fstring %) (mapv float x))))
      (format fstring (float x)))))

(defn sample-printer
  [inputs outputs targets errors]
  (let [float-formatter (partial to-string 5)
        v (mapv #(mapv float-formatter %) [inputs outputs targets errors])]
    (clojure.pprint/print-table
     (apply
      (partial mapv #(zipmap ["INPUT" "OUTPUT" "TARGET" "ERROR"]
                             (vector %1 %2 %3 %4)))
      v))))

(defn sample-output
  "Produces the output and the error for N iterations of NET under profile
  TRAINING-PROFILE"
  [net training-profile sample-count]
  (let [inputs (take sample-count (repeatedly (:input-fn training-profile)))
        outputs (mapv #(backprop/net-eval net %) inputs)
        targets (mapv (:output-fn training-profile) inputs)
        errors (mapv (error-fns/get-cost-fn (:cost-fn training-profile))
                     outputs targets)
        ;perrors (mapv #(perror %1 %2 training-profile) outputs targets)
        ]
    (sample-printer inputs outputs targets errors)))



(defn- prompt-read
  [prompt]
  (printf "%s: " prompt)
  (flush)
  (read-line))

(defn- y-or-n?
  [prompt]
  (= "y"
     (loop []
       (or
        (re-matches #"[yn]" (.toLowerCase (prompt-read prompt)))
        (do (newline)
            (recur))))))

                                        ; TODO: make it responsive so user knows trianing is continuig
(defn- continue-prompt
  [tprofile]
  (if (y-or-n? "Continue training? (y/n)")
    (do (newline)
        (printf "How many interations? (integer) ")
        (flush)
        (let [iterations (Integer. (read-line))]
          (newline)
          (if (y-or-n? (format "The current learning rate is %f. Would you like to change it? (y/n) "
                               (float (:lrate tprofile))))
            (do (newline)
                (printf "Enter a new value: ")
                (flush)
                (let [new-lrate (Float. (read-line))]
                  [iterations new-lrate]))
            [iterations (:lrate tprofile)])))
    nil))


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
      (do (sample-output next-net training-profile 20)
          (let [cont (continue-prompt training-profile)]
            (when cont
              (train-for next-net
                         (assoc training-profile :lrate (second cont))
                         (first cont)))))
      (recur next-net training-profile (dec iterations)))))

