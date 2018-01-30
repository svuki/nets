(ns nets.mnist-example
  (:require
            [nets.net :as net]
            [nets.backpropogation :as backprop]
            [nets.error-functions :as error-fns]
            [nets.printers :as printers])
  (:use nets.utils nets.matrix-utils))

;;; This file contains an example using the MNIST handwritten digit
;;; classification data set. The data is is located in external files,
;;; it may downloaded from http://yann.lecun.com/exdb/mnist/. There are four
;;; files: (1) a training set of 60,000 images, (2) labels for the training
;;; set, (3) a test set of 10,000 images, and (4) labels for the test set. The
;;; images are greyscale.

(def training-labels-stream
  (java.io.RandomAccessFile. "test/nets/train-labels.idx1-ubyte" "r"))
(def training-images-stream
  (java.io.RandomAccessFile. "test/nets/train-images.idx3-ubyte" "r"))
(def test-labels-stream
  (java.io.RandomAccessFile. "test/nets/t10k-labels.idx1-ubyte" "r"))
(def test-images-stream
  (java.io.RandomAccessFile. "test/nets/t10k-images.idx3-ubyte" "r"))


;;; The files are in the IDX file format. For our purposes the relevant data
;;; is as follows:
;;;     Label files:
;;;         Labels begin at offset 0008.
;;;         Each label is a single unisgned byte.
;;;         Label values range from 0 to 9.


(defn go-to-labels
  "Sets the filestream to the byte of the first label"
  [label-stream]
  (.seek label-stream 8))

(defn get-label
  "Returns the Nth label as an integer."
  [label-stream]
  (let [c (try (.read label-stream)
               (catch java.io.EOFException e
                 (do (go-to-labels label-stream)
                     (.read label-stream))))]
    c))

;;;     Image files:
;;;         Image data begins at offset 0016.
;;;         Each pixel is a single unsigned byte.
;;;         Pixel values range from 0 to 255.
;;;         Each image is a 28 by 28 pixels.
;;;         The data for a single image is 784 consecutive pixels values.

(defn go-to-images
  "Sets the filestream to the first pix data."
  [fstream]
  (.seek fstream 16))
(defn get-image
  "Returns the next image as a vector of 784 doubles. Each pixel value is divided
  by the maximum possible pixel value (255) to produce a number between 0 and 1."
  [fstream]
  (try  (into [] (repeatedly (* 28 28)
                             (fn [] (/ (.read fstream)
                                       255.0))))
        (catch java.io.EOFException e
          (do (go-to-images fstream)
              (get-image fstream)))))


(def mnist-training-input-fn
  #(get-image training-images-stream))
;;; For now we use these function with core/train-for. That function specified
;;; that outputs take one argument, so we have to discard the argument here.
(def mnist-training-output-fn
  (fn [_] (one-hot (get-label training-labels-stream) 10)))
(def mnist-test-input-fn
  #(get-image test-images-stream))
(def mnist-test-output-fn
  (fn [_] (one-hot (get-label test-labels-stream) 10)))

(def mnist-tprofile
  {:lrate 0.2
   :cost-fn :cross-entropy
   :input-fn mnist-training-input-fn
   :output-fn mnist-training-output-fn})

(defn prediction-vector
  [v]
  (let [mx (apply max v)]
    [(.indexOf v mx) mx]))

(defn wrong?
  [prediction-vec target]
  (let [target-index (.indexOf target 1.0)]
    (= target-index (first prediction-vec))))

(defn print-summary
  [output target]
  (let [pvec (prediction-vector output)]
    (printf "Predicts %d with confidence %.3f.\n" (first pvec) (second pvec))
    (printf "Actual: %d.\n" (.indexOf target 1.0))))

(defn printer
  [output target]
  (print-summary output target)
  (newline)
  (printers/print-vec-comp output target))

(defn mnist-sample
  [net training-profile trials]
  (let [inputs   (repeatedly trials (:input-fn training-profile))
        output  (map #(backprop/net-eval net %) inputs)
        targets (map (:output-fn training-profile) (range trials))]
    (dorun (map printer output targets))))

(defn continue-prompt
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



(defn mnist-runner
  "A training runner for mnist dataset."
  [net training-profile iterations]
  (let [input ((:input-fn training-profile))
        output ((:output-fn training-profile) input)
        next-net (backprop/train net input output
                                 (:lrate training-profile)
                                 (error-fns/get-cost-grad
                                  (:cost-fn training-profile)))]
    (when (= 0 (mod iterations 10))
      (printf "Iterations remaining: %d.\n" iterations)
      (flush))
    (if (= 0 iterations)
      (do (mnist-sample net training-profile 10)
          (let [cont (continue-prompt training-profile)]
            (if cont
              (mnist-runner next-net
                                 (assoc training-profile :lrate (second cont))
                                 (first cont))
              (do (newline)
                  (printf "OK, exiting.\n")))))
      (recur next-net training-profile (dec iterations)))))

(defn mnist-example
  [iterations]
  (let [test-net (net/new-net (* 28 28) [400 :leaky-relu] [200 :leaky-relu] [10 :softmax])]
    (mnist-runner test-net mnist-tprofile iterations)))

