(ns nets.backpropogation-test
  (:require [nets.backpropogation :as b]
            [clojure.test :as t]))


;;; In test we propogate the input [1 1] through a sample net with
;;; a simple activation function and gradient functions. The target
;;; output is [2]
(def basic-net
  "A simple net with 2 inputs, a hidden layer of size 3, and a single output.
  Uses + 1 as its activation function, and the function v -> -v for the
  gradient of the derivative."
  {:num-inputs 2
   :layers
   [{:matrix [[0.0 1.0]
              [1.0 0.0]
              [1.0 1.0]]
     :bias [1.0 -1.0 1.0]
     :act-fn (fn [v] (mapv inc v))
     :deriv-fn (fn [v] (mapv - v))}
    {:matrix [[-1.0 1.0 -1.0]]
     :bias [5.0]
     :act-fn (fn [v] (mapv inc v))
     :deriv-fn (fn [v] (mapv - v))}]})

(def basic-net-layer-activation
  "The outputs from each of the layers, including the input layer"
  [[1.0 1.0] [3.0 1.0 4.0]])
(def basic-net-layer-preactivation
  "The inputs to each of the layer's activation functions"
  [[2.0 0.0 3.0] [-1.0]])
(def basic-net-output
  "The total output of the basic net"
  [0.0])

(t/deftest fprop
  (t/testing "Inner propgate forwards function"
    (let [[v1 v2 x] (b/propogate-forward-inner basic-net [1.0 1.0])]
      (t/is (= basic-net-layer-activation v1) "Layer-outputs is incorrect")
      (t/is (= basic-net-layer-preactivation v2)
            "Preactivation-values is incorrect")
      (t/is (= basic-net-output x) "Net output is incorrect")))
  (t/testing "Outer propogate forwards function"
    (let [m (b/propogate-forward-outer basic-net [1.0 1.0])]
      (t/is (= (:layer-outputs m) basic-net-layer-activation))
      (t/is (= (:layer-preact m) basic-net-layer-preactivation))
      (t/is (= (:net-output m) basic-net-output)))))

(def bprop-vals
  "The bprop vals for the test net, assuming a target signal of [2.0]"
  [[-4.0 0.0 -6.0] [-2.0]])

(t/deftest deltas
  ; Test the error signal generation with dummy values
  (t/testing "Generating ouptut error-signal"
    (let [test-map {:layer-preact [[2.0 -1.0]] :net-output [1.0 0.0]}
          test-cost-grad (fn [v1 v2] (mapv - v1 v2))
          test-target [2.0 2.0]
          output-deriv (fn [v] (mapv #(* 0.5  %) v))]
      (t/is (= [-1.0 1.0]
               (b/error-signal test-map test-target
                               test-cost-grad output-deriv)))))
  (t/testing "Inner layer backpropogation"
    (let [fprop-map
          {:layer-outputs basic-net-layer-activation
           :layer-preact  basic-net-layer-preactivation
           :net-output    basic-net-output}
          cost-gradient (fn [actual target] (mapv - actual target))]
      (t/is (= (b/deltas basic-net fprop-map [2.0] cost-gradient)
               bprop-vals)
            "Calculates backpropogation values correctly"))))

(def deriv-matrices
  [[[-4.0 0.0 -6.0]
    [-4.0 0.0 -6.0]]
   [[-6.0]
    [-2.0]
    [-8.0]]])

(def updated-matrices
  "Assumes lrate of 0.5"
  [[[2.0 1.0 4.0]
    [3.0 0.0 4.0]]
   [[2.0]
    [2.0]
    [3.0]]])

(def updated-biases
  [[3.0 -1.0 4.0]
   [6.0]])

(t/deftest weight-update
  (t/is (= deriv-matrices
           (b/deriv-matrices basic-net-layer-activation bprop-vals))
        "Produces cost/weight deriv matrices")
  (t/is (= updated-matrices
           (b/weight-update-matrices basic-net basic-net-layer-activation
                                     0.5 bprop-vals)))
  (let [new-net (b/weight-update basic-net basic-net-layer-activation
                               bprop-vals 0.5)]
    (t/is (= updated-matrices
             (mapv :matrix (:layers new-net)))
          "Weight update updates matrices correctly")
    (t/is (= updated-biases
             (mapv :bias (:layers new-net)))
          "Weight update updates biases correctly")))
