(ns nets.core
  (:require [nets.backpropogation :as backprop]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns])
  (:gen-class))

(defn verbose-trainer
  "Will output the input, output, desired output, and error for the training example."
  [net input target error-fn deriv-error-fn learning-rate]
  (let [fvals (backprop/propogate-forward net input)
        bvals (backprop/backpropogate net fvals target deriv-error-fn)
        output (last fvals)
        error (error-fn target output)]
    (do (print "input ")
        (print input)
        (print " output ")
        (print output)
        (print " target ")
        (print target)
        (print (format " error: %.4f\n" error))
        (backprop/weight-update net fvals bvals learning-rate))))

(defn multi-set-train-verbose
  [net input-func target-func error-fn deriv-error-fn learning-rate count]
  (if (= count 0)
    net
    (let [input (input-func)
          target (target-func input)
          next-net
          (verbose-trainer net input target error-fn deriv-error-fn learning-rate)]
      (recur
       next-net
       input-func
       target-func
       error-fn
       deriv-error-fn
       learning-rate
       (dec count)))))

; TODO: generalize perror, easiest way is to have a metric selection which
; abstract away err-fn, err-deriv, and the vector size in metric
(defn perror
  "Calculates the percent error given an ERROR and a TARGET value.
  Currently this only works for the l2-metric."
  [error target]
  (let [zero-vec (take (count target) (repeat 0))]
    (/ error (error-fns/l2 target zero-vec))))

(defn sample-output
  "Produces the output and the error for N iterations of NET under profile
  TRAINING-PROFILE"
  [net training-profile sample-count]
  (let [inputs (take sample-count (repeatedly (:input-fn training-profile)))
        outputs (mapv #(backprop/net-eval net %) inputs)
        targets (mapv (:output-fn training-profile) inputs)
        errors (mapv (:err-fn training-profile) targets outputs)
        perrors (mapv perror errors targets) ]
    (clojure.pprint/print-table
     (mapv #(zipmap [:input :output :target :error :perror]
                    (vector %1 %2 %3 %4 %5))
           inputs outputs targets errors perrors))))


(defn train-for
  [net training-profile iterations]
  (let [input ((:input-fn training-profile))
        output ((:output-fn training-profile) input)
        next-net (backprop/train net input output
                                 (:lrate training-profile)
                                 (:err-deriv training-profile))]
    (if (>= 1 iterations)
      (sample-output next-net training-profile 30)
      (recur next-net training-profile (dec iterations)))))
