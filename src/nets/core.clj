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
      (multi-set-train-verbose
       next-net
       input-func
       target-func
       error-fn
       deriv-error-fn
       learning-rate
       (dec count)))))

(defn until-error-multi-set-train-verbose
  [net input-func target-func error-fn deriv-error-fn learning-rate count max-err]
  (let [input (input-func)
        target (target-func input)
        fvals (backprop/propogate-forward net input)
        bvals (backprop/backpropogate net fvals target deriv-error-fn)
        output (last fvals)
        error (error-fn target output)]
    (if (< error max-err)
      (do (printf "Completed in %d runs.\n")
          net)
      (let [new-net
            (verbose-trainer
             net input target error-fn
             deriv-error-fn learning-rate)]
        (recur new-net input-func target-func error-fn deriv-error-fn
               learning-rate (inc count) max-err)))))
