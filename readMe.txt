BoW_ORB			codes extracted from ORB_SLAM2 system, responsible for extraction of ORB features

Evaluation/Evaluation.cpp 		runs experiments described in the paper with semantic features
Evaluation/EvaluationORB.cpp	runs experiments described in the paper with ORB features
Evaluation/src/BuildVoc.cc		build BoW vocabulary from ORB features
Evaluation/src/ORBextractor.cc	extracts ORB features from images
Evaluation/src/imgCompare.cpp	matches templates extracted from two images and return a similarity score
Evaluation/src/templateExtractor.cpp	extractes templates from images

Projection/projection.cpp		build histogram matrix to calculate condiational probaiblity
Projection/ConPro.cpp		template matching based on conditional probability
