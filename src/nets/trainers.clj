(ns nets.trainers
  (:require [nets.backpropogation :as backprop]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns]
            [clojure.pprint]
            [nets.mnist-example :as mnist]
            [nets.printers :as printers])
  (:use [nets.utils])
  (:gen-class))

(defn- sample-output
  "Produces the output and the error for N iterations of NET under profile
  TRAINING-PROFILE"
  [net training-profile sample-count]
  (let [inputs (take sample-count (repeatedly (:input-fn training-profile)))
        outputs (mapv #(backprop/net-eval net %) inputs)
        targets (mapv (:output-fn training-profile) inputs)
        errors (mapv (error-fns/get-cost-fn (:cost-fn training-profile))
                     outputs targets)]
    (printers/sample-printer inputs outputs targets errors)))


(defn- continue-prompt
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
  "Runs ITERATIONS iterations of stochastic gradient descent starting wtih net NET
  under the training profile TRAINING-PROFILE (this contains (1) the input function to be
  used, (2) the target function, (3) the learning rate, and (4) the cost function. Upon
  completion, train-for will call prompt the user to continue training. If they say yes, they have an
  opportuninty to change the hyperparameters and number of iterations before trainng resumes."
  [net training-profile iterations]
  (let [input ((:input-fn training-profile))
        output ((:output-fn training-profile) input)
        next-net (backprop/sgd net input output
                                 (:lrate training-profile)
                                 (error-fns/get-cost-grad
                                  (:cost-fn training-profile)))]
    (if (>= 1 iterations)
      ; TODO: hardcoded 20, see if this part can be passed in as a higher order function to modify
      ; sampling behaviour
      (do (sample-output next-net training-profile 20)
          (let [cont (continue-prompt training-profile)]
            (when cont
              (train-for next-net
                         (assoc training-profile :lrate (second cont))
                         (first cont)))))
      (recur next-net training-profile (dec iterations)))))

(defn- notify-completion
  []
  (println "The computation has completed."))

(defn sgd-handler
  "Runs SGD in a seperate thread, allowing the user to query the current state of the computation."
  [initial-net tprofile iterations]
  {:pre [(pos? iterations)]}
  (let [net         (agent initial-net)
        time-est    (promise)
        iters       (agent iterations)
        proc        (fn [] (send net backprop/sgd tprofile) (send iters dec))
        computation
        (future
          (do
              (if (< iterations 50)
                (deliver time-est nil)
                (deliver time-est
                         (time-sec (dorun (repeatedly 50 proc)))))
              (dorun (repeatedly @iters proc))
              (println "The computation has completed.")))]
    (fn [kw & args]
      (case kw
        :net        @net ;return the current value of net
        :done?      (future-done? computation)
        :save       (apply net/to-file args)
        :iterations @iters
        :time       (if (and (realized? time-est) @time-est)
                      (printf "Estimated time remaining: %.2f seconds.\n"
                              (* @time-est (/ @iters 50.0)))
                      (println "Still Estiming...\n"))
        (println "Invalid option. Options are :net, :done?, :save, :iterations, :time")))))
