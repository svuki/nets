(defproject nets "0.1.0-SNAPSHOT"
  :description "Stochastic gradient descent for multi layer perceptrons"
  :url "https://github.com/svuki/nets"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/algo.generic "0.1.2"]
                 [net.mikera/core.matrix "0.61.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [clatrix "0.5.0"]
                 [net.mikera/imagez "0.12.0"]
                 [uncomplicate/neanderthal "0.18.0"]
                 [uncomplicate/fluokitten "0.6.1"]]
  :main ^:skip-aot nets.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
