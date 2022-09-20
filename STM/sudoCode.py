#calculate similarity between frame1 and frame2
compare(Frame1, Frame2)

# calculate similarity score between the current frame and each of its connected frame,
# return the minimum similarity score
get_minScore(cFrame, conFrames)

#choose a frame from scene 's' in 'l' image set
get_cFrame(s,l)

#choose connected frames from scene 's' in 'l' image set
get_conFrames(s,l, cFrame)

#choose all images from scene 's' in lighting condition 'l', 
# then choose all images from some randomly selected scenes in lighting condition 'l'
#returns [[img_s,img_s,img_s],[img_s1, img_s1, img_s1],[img_s2, img_s2, img_s2]]
#note the first list is the true list for loop closure
get_loopCandidates(l,s)

#sample a few images from each list in loopCandidate_lists, calculate similarity between 
#each of these images and cFrame, if similarity is higher than minScore, add the list the image in
#to filteredCandidates
filterCandidates(loopCandidate_lists, minScore)

S=[s1, s2, s3] #list of scenes
L=[l1, l2, l3] #list of lighting conditions

Experiment 1:

for s in S:
    cFrame = get_cFrame(s,ref) #get current frame from images of scene 's' in reference imageset
    conFrames = get_conFrames(s,ref, cFrame) #get frames connected to current frame
    minScore = get_minScore(cFrame, conFrames) #get minimum similarity score between cFrame and each of its connected frames
    for l in L:
        loopCandidate_lists = get_loopCandidates(l,s)
        loopCandidate_filtered = filterCandidates(loopCandidate_lists, minScore)
        
        score_list=[] #similarity score for each list in loopCandidate_list
        for loopList in loopCandidate_filtered:
            list_score=0
            maxScore=0
            for loopFrame in loopList:
                score = compare(cFrame,loopFrame) 
                list_score+=score
            if(list_score>maxScore):
                maxScore=list_score
            score_list.append(list_score)

        #loop closure is considered as found if the score for true loop 
        #closure list is the highest among all the other lists 
        if(score_list[0]=maxScore): 
            loopFound = true

        

