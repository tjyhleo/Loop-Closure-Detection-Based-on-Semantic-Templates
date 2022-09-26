#include "imgCompare.h"


//find a group of templates that have the same labels as input template and are closest to input template
bool findGroup(vector<vector<int>>& tempGroup, vector<vector<int>> allTemplates, vector<int>targetLabels){
    bool groupFound=false; //if a group of templates are found
    vector<int> mostClose; //the template that's closet to input template
    vector<int> secondClose; //the template that's second close to input template
    float minD, minD2; //minD distance between mostClose and input template, minD2 distance btw secondClose and input template
    size_t inputIdx = tempGroup[0][2]; //index of input template
    vector<int>inputP{tempGroup[0][0], tempGroup[0][1]}; //x,y of input template
    for(size_t st=0; st<allTemplates.size(); st++){
        //if the current template is the input template, move on
        if(st==inputIdx){
            continue;
        }
        vector<int>Labels{allTemplates[st][2],allTemplates[st][3]}; //labels of current template
        //if current template label is different from the input template, move on
        if(Labels!=targetLabels){
            continue;
        }
        vector<int>currentP{allTemplates[st][0],allTemplates[st][1]}; //x,y of current template
        float distance = norm(inputP, currentP);
        //if this is the first template, mark it as mostClose
        if(mostClose.empty()){
            mostClose=currentP;
            mostClose.push_back(int(st));
            mostClose.push_back(allTemplates[st][4]);
            minD=distance;
        }
        else if(distance<minD){
            secondClose=mostClose;
            minD2=minD;
            mostClose=currentP;
            mostClose.push_back(int(st));
            mostClose.push_back(allTemplates[st][4]);
            minD=distance;
        }
        else if(secondClose.empty()){
            secondClose=currentP;
            secondClose.push_back(int(st));
            secondClose.push_back(allTemplates[st][4]);
            minD2=distance;
        }
        else if(distance<minD2){
            secondClose=currentP;
            secondClose.push_back(int(st));
            secondClose.push_back(allTemplates[st][4]);
            minD2=distance;
        }
    }
    if(!mostClose.empty() && !secondClose.empty()){
        groupFound=true;
        tempGroup.push_back(mostClose);
        tempGroup.push_back(secondClose);
    }
    
    return groupFound;
}

//match groups
bool matchGroup(vector<vector<int>>qryGroup, vector<vector<int>>refGroup){
    bool matched=false;
    float thre=0.1;
    int qryF1 = qryGroup[0][3];
    int qryF2 = qryGroup[1][3];
    int qryF3 = qryGroup[2][3];
    int refF1 = refGroup[0][3];
    int refF2 = refGroup[1][3];
    int refF3 = refGroup[2][3];
    if(qryF1!=refF1 || qryF2!=refF2 || qryF3!=refF3){
        return matched;
    }

    vector<int>qryP1{qryGroup[0][0],qryGroup[0][1]};
    vector<int>qryP2{qryGroup[1][0],qryGroup[1][1]};
    vector<int>qryP3{qryGroup[2][0],qryGroup[2][1]};
    vector<int>refP1{refGroup[0][0],refGroup[0][1]};
    vector<int>refP2{refGroup[1][0],refGroup[1][1]};
    vector<int>refP3{refGroup[2][0],refGroup[2][1]};
    float qryD1=norm(qryP1, qryP2);
    float qryD2=norm(qryP1, qryP3);
    float qryD3=norm(qryP2, qryP3);
    float refD1=norm(refP1, refP2);
    float refD2=norm(refP1, refP3);
    float refD3=norm(refP2, refP3);
    float qryR1=qryD1/qryD2;
    float qryR2=qryD1/qryD3;
    float qryR3=qryD2/qryD3;
    float refR1=refD1/refD2;
    float refR2=refD1/refD3;
    float refR3=refD2/refD3;
    float diffQR1=abs(qryR1-refR1)/refR1;
    float diffQR2=abs(qryR2-refR2)/refR2;
    float diffQR3=abs(qryR3-refR3)/refR3;

    if(diffQR1<thre && diffQR2<thre && diffQR3<thre){
        matched=true;
    }


    return matched;
}

int coarseMatch(vector<vector<int>> qryTemplates, vector<vector<int>> refTemplates){
    int matches=0;
    for(size_t st=0; st<qryTemplates.size(); st++){
        vector<int> qryTemplate=qryTemplates[st];
        //if this template has been matched, move on
        if(qryTemplate[4]==1){
            continue;
        }
        vector<int> qryLabels{qryTemplate[2], qryTemplate[3]};//semantic labels of current template
        //find a group of templates close to the current template
        vector<vector<int>>qryGroup;//a group of close query templates, [[x,y,idx,flag],[x,y,idx,flag],[x,y,idx, flag]]
        vector<int> xyIdx{qryTemplate[0],qryTemplate[1],int(st), qryTemplate[4]};
        qryGroup.push_back(xyIdx);
        bool groupFound = findGroup(qryGroup, qryTemplates, qryLabels);
        //if we cannot find a group for this query template, move on
        if(groupFound==false){
            continue;
        }

        for(size_t r=0; r<refTemplates.size(); r++){
            vector<int>refTemplate=refTemplates[r];
            //if this template has been matched, move on
            if(refTemplate[4]==1){
                continue;
            }
            //if the reference template label is not the same as the query template label, move on
            vector<int>refLabels{refTemplate[2],refTemplate[3]};
            if(refLabels!=qryLabels){
                continue;
            }
            //find a group of templates close to the current template
            vector<vector<int>>refGroup;//a group of close ref templates, [[x,y,idx],[x,y,idx],[x,y,idx]]
            vector<int> xyIdx{refTemplate[0],refTemplate[1],int(r), refTemplate[4]};
            refGroup.push_back(xyIdx);
            bool groupFound = findGroup(refGroup, refTemplates, refLabels);
            //if we cannot find a group for this reference template, move on
            if(groupFound==false){
                continue;
            }
            //match the refGroup and qryGroup
            bool groupMatched = matchGroup(qryGroup, refGroup);
            //if the groups are matched, turn the flag of all templates in both groups to 1
            if(groupMatched==true){
                // cout<<"group is matched"<<endl;
                for(size_t tt=0; tt<qryGroup.size(); tt++){
                    int idx = qryGroup[tt][2];
                    qryTemplates[idx][4]=1;
                }
                for(size_t tt=0; tt<refGroup.size(); tt++){
                    int idx = refGroup[tt][2];
                    refTemplates[idx][4]=1;   
                }
                break;
            }
        }
    }
    //count the number of templates matched in query templates
    for(size_t st=0; st<qryTemplates.size(); st++){
        if(qryTemplates[st][4]==1){
            matches+=1;
        }
    }
    
    for(size_t st=0; st<refTemplates.size(); st++){
        if(refTemplates[st][4]==1){
            matches+=1;
        }
    }

    return matches;
}

int fineMatch(vector<vector<int>> qryTemplates, vector<vector<int>> refTemplates){
    int matches=0;
    for(size_t qst=0; qst<qryTemplates.size(); qst++){
        vector<int> qryLabels{qryTemplates[qst].begin()+3,qryTemplates[qst].end()};
        for(size_t rst=0; rst<refTemplates.size(); rst++){
            //if this template has been matched, move on
            if(refTemplates[rst][2]==1){
                continue;
            }
            vector<int> refLabels{refTemplates[rst].begin()+3,refTemplates[rst].end()};
            if(qryLabels==refLabels){
                qryTemplates[qst][2]=1;
                refTemplates[rst][2]=1;
                break;
            }
        }
    }
    
    //sum up the number of template matches
    for(size_t st=0; st<qryTemplates.size(); st++){
        if(qryTemplates[st][2]==1){
            matches+=1;
        }
    }
    for(size_t st=0; st<refTemplates.size(); st++){
        if(refTemplates[st][2]==1){
            matches+=1;
        }
    }

    return matches;
}

float imgCompare(vector<vector<int>> fineTemplates1,
                vector<vector<int>> coarseTemplates1,
                vector<vector<int>> fineTemplates2,
                vector<vector<int>> coarseTemplates2)
{
    int T1 = fineTemplates1.size()+coarseTemplates1.size();
    int T2 = fineTemplates2.size()+coarseTemplates2.size();
    int fineNum = fineMatch(fineTemplates1, fineTemplates2);
    int coarseNum = coarseMatch(coarseTemplates1, coarseTemplates2);
    float output = float(fineNum+coarseNum)/float(T1+T2);
    return output;
}