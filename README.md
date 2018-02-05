# WalkRanker
Java implementation of item recommendation algorithm WalkRanker. Thanks to the LibRec project. This repository reuses some core components from LibRec version 1.2. We implements asynchronous SGD to speed up model learning. 

Input format: userid \t itemid \t rating

You need split traning and testing by yourself, then implement java command as follows. 

"java -jar walkranker.jar -train data/ml100k/u1.base -test data/ml100k/u1.test"

If you wanna reuse this project, please import this repository as existing project.
