(ns nets.core
  (:require [nets.backpropogation :as backprop]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns]
            [clojure.pprint]
            [nets.mnist-example :as mnist]
            [nets.printers :as printers])
  (:use [nets.utils])
  (:gen-class))

(defn sample-output
  "Produces the output and the error for N iterations of NET under profile
  TRAINING-PROFILE"
  [net training-profile sample-count]
  (let [inputs (take sample-count (repeatedly (:input-fn training-profile)))
        outputs (mapv #(backprop/net-eval net %) inputs)
        targets (mapv (:output-fn training-profile) inputs)
        errors (mapv (error-fns/get-cost-fn (:cost-fn training-profile))
                     outputs targets)]
    (printers/sample-printer inputs outputs targets errors)))


(defn continue-prompt
  "Prompt the user if they'd like to continue training. If they affirm, offers to change
  the learning rate."
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


;;;;;;;;;;;;;;;;;

(defn -main
  [ iterations ]
  (mnist/mnist-example (Integer. iterations)))
