/*
  _____  _____ ______ _  _    ___   ___ ___
 / ____|/ ____|  ____| || |  / _ \ / _ \__ \
| |    | (___ | |__  | || |_| | | | (_) | ) |
| |     \___ \|  __| |__   _| | | |> _ < / /
| |____ ____) | |____   | | | |_| | (_) / /_
 \_____|_____/|______|  |_|  \___/ \___/____|
                                            
 ARTIFICIAL INTELLIGENCE
    Homework-1
 
 Author: ONURCAN ISLER
 */

#include <vector>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <ios>
#include <fstream>
#include <string>
#include <cmath>
#include <stack>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
// To get the memory size of macOS, I used library below:
// #include <mach/mach.h>
// However, I will comment the parts where we compute memory size when submitting this homework.
// Because, most probably your operating system will not work well with this mach.h functions.
using namespace std;

class Node{
public:
    shared_ptr<Node> parent; // shared_ptr kinda works like a garbage collector.
    vector<vector<bool>>board; // Each state is a board.
    Node(vector<vector<bool>> rhsBoard){
        this->board = rhsBoard;
    }
};


// **************************
// *                        *
// *    COMMON VARIABLES    *
// *                        *
// **************************
// These variables below are all common for each method BFS, DFS, IDS etc.
// We separate these variables to improve readability of the code.

// Slots would help us to determine which nodes to remove first.
// We need slots table because of the condition in the document:
    // For methods a to c, if there are multiple children to be put inside the
    // frontier list, put the children in such an order that the child node with
    // the smallest numbered peg is removed from the board is selected first.
vector<vector<int>>slots = {
{-1, -1, 1, 2, 3, -1, -1},
{-1, -1, 4, 5, 6, -1, -1},
{7, 8, 9, 10, 11, 12, 13},
{14, 15, 16, 17, 18, 19, 20},
{21, 22, 23, 24, 25, 26, 27},
{-1, -1, 28, 29, 30, -1, -1},
{-1, -1, 31, 32, 33, -1, -1}
};

int numberOfExpandedNodes = 0;
// Expanded node is the one that inserted into frontier.
int maxNumberOfStoredNodes = 0;
shared_ptr<Node> bestNode = NULL;
// Keep track of the best node i.e. best state found so far.
int n = 7;
// Number of rows = Number of columns = 7
double maxmemoryusage = 0.0;



// ************************
// *                      *
// *    COMMON METHODS    *
// *                      *
// ************************
// These methods are all common for each method BFS, DFS, IDS etc.
// We separate these functions to improve readability of the code.


vector<vector<bool>> initializeBoard(){
    // Initialize the board for each playing method.
    numberOfExpandedNodes = 0;
    maxNumberOfStoredNodes = 0;
    bestNode = NULL;
    maxmemoryusage = 0.0;
    
    vector<vector<bool>>board;
    
    board.push_back({0,0,1,1,1,0,0});
    board.push_back({0,0,1,1,1,0,0});
    board.push_back({1,1,1,1,1,1,1});
    board.push_back({1,1,1,0,1,1,1});
    board.push_back({1,1,1,1,1,1,1});
    board.push_back({0,0,1,1,1,0,0});
    board.push_back({0,0,1,1,1,0,0});
    
    return board;
}

void printBoard(vector<vector<bool>>&board){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(slots[i][j]==-1){
                cout << ' ';
            }
            else if(board[i][j]==0){
                cout << '.';
            }
            else{
                cout << 'O';
            }
        }
        cout << endl;
    }
}

// To remove the pegs we use following functions.
// Here for example:
//     ...
//     ...
//   .......
//   .......
//   ...0OOO
//     ...
//     ...
// the peg 0 is horizontally removable. Since its left is free and there is another peg on right.
bool isRemovableVertically(const int&i, const int&j, vector<vector<bool>>&board){
    if(board[i][j]==0){return false;}
    // If both board[i+1][j] and board[i-1][j] are 1 or 0, then we have up and down pegs.
    // So, we can not eat vertically.
    if(i+1<n && i>0 && slots[i+1][j]!=-1 && slots[i-1][j]!=-1 && ((board[i+1][j]^board[i-1][j])==1)){
        return true;
    }
    return false;
}
bool isRemovableHorizontally(const int&i, const int&j, vector<vector<bool>>&board){
    if(board[i][j]==0){return false;}
    // If both board[i][j+1] and board[i][j-1] are 1 or 0, then we right and left pegs.
    // So, we can not eat horizontally.
    if(j+1<n && j>0 && slots[i][j+1]!=-1 && slots[i][j-1]!=-1 && ((board[i][j+1]^board[i][j-1])==1)){
        return true;
    }
    return false;
}
// Removable if we can eat it vertically or horizontally.
bool isRemovable(const int&i, const int&j, vector<vector<bool>>&board){
    return isRemovableVertically(i, j, board) || isRemovableHorizontally(i, j, board);
}

// Traverse the board and return the indices of the pegs that are removable.
vector<vector<int>> getRemovablePegs(vector<vector<bool>>&board){
    vector<vector<int>> result;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(board[i][j]==0 || slots[i][j]==-1){continue;}
            
            if(isRemovable(i, j, board)){
                result.push_back({i,j});
            }
        }
    }
    return result;
}

// Here sort the removable pegs so that:
// the pegs with smallest slot numbers would be picked first.
// We sort in reverse order and return it.
vector<vector<int>> getOrderedRemovablePegs(vector<vector<bool>>&board){
    vector<vector<int>>removablePegs = getRemovablePegs(board);
    vector<vector<int>>helpToSort(removablePegs.size());
    for(int i=0;i<removablePegs.size();i++){
        // Insert slot numbers as first index. It will help us to sort pegs.
        helpToSort[i].push_back(slots[removablePegs[i][0]][removablePegs[i][1]]);
        helpToSort[i].push_back(removablePegs[i][0]);
        helpToSort[i].push_back(removablePegs[i][1]);
        
    }
    // Sort the removable pegs so that the pegs with smallest slot numbers
    // will be on the top.
    sort(helpToSort.begin(),helpToSort.end());
    for(int i=0;i<removablePegs.size();i++){
        removablePegs[i][0] = helpToSort[i][1];
        removablePegs[i][1] = helpToSort[i][2];
    }
    //reverse(removablePegs.begin(),removablePegs.end());
    return removablePegs;
}

// Count how many pegs left on the board.
int countPegs(vector<vector<bool>>&board){
    int count = 0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(slots[i][j]==-1){continue;}
            count += board[i][j];
        }
    }
    return count;
}

// Goal state when there is a single peg left in the middle of the board.
bool isGoalState(vector<vector<bool>>&board){
    if(countPegs(board)==1 && board[3][3]==1){
        return true;
    }
    return false;
}

// Print the solution path of the bestNode.
void printPath(shared_ptr<Node>root){
    shared_ptr<Node> curr = root;
    vector<shared_ptr<Node>> visitedNodes;
    while(curr!=NULL){
        visitedNodes.push_back(curr);
        curr = curr->parent;
    }
    reverse(visitedNodes.begin(),visitedNodes.end());
    for(int i=0;i<visitedNodes.size();i++){
        cout << i+1 << ". state:" << endl;
        printBoard(visitedNodes[i]->board);
    }
}

// We already know which pegs to remove. So, it is time to remove them.
// We can easly remove a peg. All we do is the XOR the immediate elements with 1.
// That way, empty cell will be filled and filled cell be emptied.
// See:
//     ...
//     ...
//   ...0...
//   ...1...
//   ...1...
//     ...
//     ...
// Here slot with board[2][3]=0 is empty. Ignore the . characters for a moment.
// We XOR board[2][3] with one, we get board[2][3]=1 which is now filled.
//     ...
//     ...
//   ...1...
//   ...1...
//   ...1...
//     ...
//     ...
// We also XOR board[2][4] with one, so that the peg will be moved.
//     ...
//     ...
//   ...1...
//   ...1...
//   ...0...
//     ...
//     ...
// Now at the end we set board[3][3]=0 since it is eaten.
//     ...
//     ...
//   ...1...
//   ...0...
//   ...0...
//     ...
//     ...
// Done! The move is completed.
vector<shared_ptr<Node>> getBoardsAfterPegsRemoved(vector<vector<int>>&removablePegs,vector<vector<bool>>&board,shared_ptr<Node>root){
    vector<shared_ptr<Node>> newBoards;
    for(int r=0;r<removablePegs.size();r++){
        int i = removablePegs[r][0];
        int j = removablePegs[r][1];
        
        
        if(isRemovableHorizontally(i, j, board)){
            vector<vector<bool>>newBoard1 = board;
            newBoard1[i][j+1] = newBoard1[i][j+1]^1; // . O O  becomes  O O .
            newBoard1[i][j-1] = newBoard1[i][j-1]^1; // O O .  becomes  . O O
            // So, end points are XORed with 1 :)
            // The same applies for the vertical removal.
            
            newBoard1[i][j] = 0;
            auto newNode = make_shared<Node>(newBoard1); // Keep the new board.
            newNode->parent = root;
            newBoards.push_back(newNode);
        }
        if(isRemovableVertically(i, j, board)){
            vector<vector<bool>>newBoard2 = board;
            newBoard2[i+1][j] = newBoard2[i+1][j]^1;
            newBoard2[i-1][j] = newBoard2[i-1][j]^1;
            
            newBoard2[i][j] = 0;
            auto newNode = make_shared<Node>(newBoard2); // Keep the new board.
            newNode->parent = root;
            newBoards.push_back(newNode);
        }
    }
    return newBoards; // Return all new boards found.
}

// At each iteration, we have check whether the specified time limit is exceed or
// the goal state is found.
bool doWeStop(shared_ptr<Node>curr,const std::chrono::steady_clock::time_point&stop){
    if(isGoalState(curr->board)){ // Goal state is found. It is time to stop.
        bestNode = curr;
        return true;
    }
    if(chrono::duration_cast<std::chrono::milliseconds>(stop - chrono::steady_clock::now()).count() < 0){
        // The specified time is elapsed. It is time to stop.
        return true;
    }
    return false; // Nothing found. Keep iterating.
}

// Suboptimal solution is a solution where no remavable pegs left.
// i.e. where we can not perform any moves anymore.
void updateSubOptimalSolution(shared_ptr<Node>curr){
    if(bestNode==NULL || (countPegs(bestNode->board)>countPegs(curr->board))){
        bestNode = curr;
    }
}

// Print the statistics that are asked in the document.
void printResults(const std::chrono::steady_clock::time_point&start, string methodName, double minutes){
    if(maxmemoryusage>500.0){
        if(bestNode!=NULL){
            printPath(bestNode);
            cout << "Program has terminated early - Memory Limit (500MB)" << endl;
            cout << "Sub-optimum Solution Found with " << countPegs(bestNode->board) << " remaining pegs." << endl;
        }
        else{
            cout << "Program has terminated early - Memory Limit (500MB)" << endl;
        }
    }
    else if(bestNode==NULL){
        cout << "No solution found – Time Limit" << endl;
    }
    else if(isGoalState(bestNode->board)){
        printPath(bestNode);
        cout << "Optimum solution found." << endl;
    }
    else{
        printPath(bestNode);
        cout << "Sub-optimum Solution Found with " << countPegs(bestNode->board) << " remaining pegs." << endl;
    }
    auto runtime = std::chrono::duration_cast<std::chrono::seconds>(chrono::steady_clock::now() - start).count();
    cout << "Search Method: " << methodName << "." << endl;
    cout << "Time Limit: " << minutes << " minutes." << endl;
    cout << "Runtime: " << ((double)runtime) / 60.0 << " minutes or " << ((double)runtime) << " seconds." << std::endl;
    cout << "Number of expanded nodes during the search: " << numberOfExpandedNodes << endl;
    cout << "Maximum number of nodes that stored in the memory at any moment: " << maxNumberOfStoredNodes << endl;
    //cout << "Maximum memory size used by the program: " << maxmemoryusage << " MB." << endl;
    cout << "=====================================================================" << endl << endl;
}

// Shuffle the pegs i.e. shuffle the children states.
// That way, DFS would pick next children randomly.
void shufflePegsRandom(vector<vector<int>>&removablePegs){
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (removablePegs.begin(), removablePegs.end(), std::default_random_engine((unsigned int)seed));
}

// Get memory size every two seconds.
// I will comment this part of the code because it only works in macOS operating systems.
int printMemoryInMB(int&numOfVisited){
    return 1;
    // I commented out the below code. It only works for macOS.
    
    /*
    if(numOfVisited%30000!=0){return -1;} // Return -1 when we don't need to print memory usage.
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(),
                                  TASK_BASIC_INFO,
                                  (task_info_t)&t_info,
                                  &t_info_count))
    {
        return -1; // We don't print memory usage.
    }
    double memsizemb = t_info.resident_size;
    memsizemb /= 1024.0;
    memsizemb /= 1024.0;
    if(maxmemoryusage<memsizemb){
        maxmemoryusage = memsizemb;
    }
    if(memsizemb>500.0){
        return -2; // Return -2 if the memory limit exceed.
    }
    //cout << "Current Memory usage:" << memsizemb << "MB" << endl;
    return 1;
    */
}


//=======================================================
//   __  __ ______ _______ _    _  ____  _____   _____
//  |  \/  |  ____|__   __| |  | |/ __ \|  __ \ / ____|
//  | \  / | |__     | |  | |__| | |  | | |  | | (___
//  | |\/| |  __|    | |  |  __  | |  | | |  | |\___ \
//  | |  | | |____   | |  | |  | | |__| | |__| |____) |
//  |_|  |_|______|  |_|  |_|  |_|\____/|_____/|_____/
//
//=======================================================
// All the below methods are sharing the same structure that is TREE-SEARCH
// algorithm shared in the slides.
// We could have combined all these methods DFS, BFS, IDS etc. into a single function by
// using a priority queue with customer comperators but in this case we would lose readability
// of the code. Since it is not explicitly stated in the homework document, we separate these
// functions.

// All functions share the same form that is below:
/*
 
 function TREE-SEARCH(problem, strategy) returns a solution, or failure
    initialize the frontier(nodes to be visited first) using the initial state of problem
    loop do
        if the frontier is empty then return failure
        choose a leaf node for expansion according to the strategy
        remove it froom it from frontier
        
        if the node contains a goal state then return the corresponding solution
        expand the node and add the resulting nodes to the frontier
    end
 
 */

void BFS(double minutes){
    auto start = chrono::steady_clock::now();
    auto stop = chrono::steady_clock::now() + chrono::seconds((int)(60.0*minutes));
    
    queue<shared_ptr<Node>>frontier;
    frontier.push(make_unique<Node>(initializeBoard())); // Intialize the frontier
    
    while(frontier.size()){
        shared_ptr<Node> curr = frontier.front(); // Choose a leaf node according the strategy i.e. FIFO.
        frontier.pop();
        numberOfExpandedNodes++;
        
        if(doWeStop(curr,stop)||printMemoryInMB(numberOfExpandedNodes)==-2){ // Check if current node contains a goal state.
            printResults(start,"Breadth-First Search",minutes); // If yes then terminate.
            return;
        }
        
        vector<vector<int>> removablePegs = getOrderedRemovablePegs(curr->board); // Find children of the current node.
        vector<shared_ptr<Node>> childrenBoards = getBoardsAfterPegsRemoved(removablePegs, curr->board, curr);
        
        if(childrenBoards.empty()){ // Is this a sub-optimal solution?
            updateSubOptimalSolution(curr); // Keep the best sub-optimal solution.
        }
        
        for(auto child:childrenBoards){ // Expand the node and add the resulting children nodes to the frontier.
            frontier.push(child);
            maxNumberOfStoredNodes = max((int)frontier.size(),maxNumberOfStoredNodes);
        }
    }
}

void DFS_Standard(double minutes){
    auto start = chrono::steady_clock::now();
    auto stop = chrono::steady_clock::now() + chrono::seconds((int)(60.0*minutes));
    
    stack<shared_ptr<Node>>frontier;
    frontier.push(make_unique<Node>(initializeBoard()));
    
    while(frontier.size()){
        shared_ptr<Node> curr = frontier.top(); // The strategy now is LIFO.
        frontier.pop();
        numberOfExpandedNodes++;
        
        if(doWeStop(curr,stop)||printMemoryInMB(numberOfExpandedNodes)==-2){ // Same functions are used. Stop if the goal stated is reached.
            printResults(start,"Depth-First Search",minutes);
            return;
        }
        
        vector<vector<int>> removablePegs = getOrderedRemovablePegs(curr->board); // Finding children nodes are also same.
        reverse(removablePegs.begin(),removablePegs.end()); // Reverse the removable pegs so that smallest node will pushed last.
        vector<shared_ptr<Node>> childrenBoards = getBoardsAfterPegsRemoved(removablePegs, curr->board, curr);
        
        
        if(childrenBoards.empty()){ // Is this a sub-optimal solution?
            updateSubOptimalSolution(curr); // Keep the best sub-optimal solution.
        }
        
        for(auto child:childrenBoards){
            frontier.push(child);
            maxNumberOfStoredNodes = max((int)frontier.size(),maxNumberOfStoredNodes);
        }
    }
}

void IDS(double minutes){
    auto start = chrono::steady_clock::now();
    auto stop = chrono::steady_clock::now() + chrono::seconds((int)(60.0*minutes));
    
    vector<vector<bool>> startBoard = initializeBoard();
    
    for(int depthLimit=1;depthLimit<35;depthLimit++){ // Depth can not be larger than 33 since we have 34 pegs.
        // We apply DFS for each depth limit 1,2,3, ... , 34.
        // Frontier is stack i.e. LIFO.
        stack<pair<int,shared_ptr<Node>>>frontier;
        frontier.push({1,make_unique<Node>(startBoard)});
        
        while(frontier.size()){
            auto curriter = frontier.top(); // Strategy is LIFO.
            frontier.pop();
            numberOfExpandedNodes++;
            
            shared_ptr<Node> curr = curriter.second;
            int currdepth = curriter.first;
            
            if(doWeStop(curr,stop)||printMemoryInMB(numberOfExpandedNodes)==-2){
                printResults(start,"Iterative Deepening Search",minutes);
                return;
            }
            
            vector<vector<int>> removablePegs = getOrderedRemovablePegs(curr->board);
            vector<shared_ptr<Node>> childrenBoards = getBoardsAfterPegsRemoved(removablePegs, curr->board, curr);
            
            if(childrenBoards.empty()){ // Is this a sub-optimal solution?
                updateSubOptimalSolution(curr); // Keep the best sub-optimal solution.
            }
            
            if(currdepth==depthLimit){ // The difference point of IDS comes here.
                continue; // Do not expand this node since it has reached depth limit.
            }
            
            for(auto child:childrenBoards){
                frontier.push({currdepth+1,child});
                maxNumberOfStoredNodes = max((int)frontier.size(),maxNumberOfStoredNodes);
            }
        }
    }
}

void DFS_RandomSelection(double minutes){
    // Almost same as DFS. The only difference is that we shuffle the children
    // before we insert them into the frontier.
    auto start = chrono::steady_clock::now();
    auto stop = chrono::steady_clock::now() + chrono::seconds((int)(60.0*minutes));
    
    stack<shared_ptr<Node>>frontier;
    frontier.push(make_unique<Node>(initializeBoard()));
    
    while(frontier.size()){
        shared_ptr<Node> curr = frontier.top();
        frontier.pop();
        numberOfExpandedNodes++;
        
        if(doWeStop(curr,stop)||printMemoryInMB(numberOfExpandedNodes)==-2){
            printResults(start,"Depth-First Search with Random Selection",minutes);
            return;
        }
        
        vector<vector<int>> removablePegs = getOrderedRemovablePegs(curr->board);
        
        shufflePegsRandom(removablePegs); // The differnece of the randomized DFS is here.
        // Shuffling the children...
        
        vector<shared_ptr<Node>> childrenBoards = getBoardsAfterPegsRemoved(removablePegs, curr->board, curr);
        
        if(childrenBoards.empty()){ // Is this a sub-optimal solution?
            updateSubOptimalSolution(curr); // Keep the best sub-optimal solution.
        }
        
        for(auto child:childrenBoards){
            frontier.push(child);
            maxNumberOfStoredNodes = max((int)frontier.size(),maxNumberOfStoredNodes);
        }
    }
}


// Here, in our project we got help from pagoda matrices to calculate heuristic values.
// But pagoda matrices are themselves not enough. We used combined tuples of heuristic values.
// The tuple we use is as follows:
// (costsum,mansum,numofpegs,visittime)
// Where,
// The first index is pagoda matrix cost sum,
// The second index manhattan distance of all pegs from the center,
// The third index number of pegs left on the board,
// The visited count of the node.

// So, how do we compare the tuples?
// See,
// (1,3,2,6) tuple1 of Node1
// (1,3,5,5) tuple2 of Node2
// Here, tuple1 is smaller and better than tuple2.
// Their pagoda matrix cost sum and manhattan distance sum are equal but...
// The number of pegs left on these boards are different.
// Since tuple1 has fewer pegs 2<5, heuristic value of Node1 is smaller and Node1.
// In this case Node1 must be picked before Node2.

// What about pagoda matrices? What are they?
// Well, in our goal state there is single peg on the center. Right?
// In this case, **we give penalty to the boards which contain many pegs on corners.**
// The last centence is extremely important.
// So, whole idea is to punish the boards with pegs farther away from center.
// But, how much?
// In this case there is nothing left but to try.
// We basically started by generating a matrix as follows:
/*
 vector<vector<int>> costpagoda = {
 { 0, 0, 1, 1, 1, 0, 0 },
 { 0, 0, 1, 0, 1, 0, 0 },
 { 1, 1, 1, 0, 1, 1, 1 },
 { 1, 0, 0, 0, 0, 0, 1 },
 { 1, 1, 1, 0, 1, 1, 1 },
 { 0, 0, 1, 0, 1, 0, 0 },
 { 0, 0, 1, 1, 1, 0, 0 }};
 */
// It was working fine and able to find boards where there was one peg left close
// to corner point. But these were not optimal solutoins. Peg were close to center
// but not exactly at the center. So, we tried many other matrices and found a one
// as follows:
int countcalls = 0;
vector<vector<int>> costpagoda = {
{ 0, 0, 4, 0, 4, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0 },
{ 4, 0, 3, 0, 3, 0, 4 },
{ 0, 0, 0, 1, 0, 0, 0 },
{ 4, 0, 3, 0, 3, 0, 4 },
{ 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 4, 0, 4, 0, 0 }};
// But these 4 and 3 values were there at first. These were basically ones.
// But then we decided change this weigts and here they are! Working perfectly fine.
// It does most of the job but not enough. In most cases boards would have same
// pagoda matrix cost. At this point tuples of manhattan distance sum etc. helped out.
/*
 WE STRONGLY BELIEVE THAT OUR HEURISTIC FUNCTION BY FAR THE BEST HEURISTIC FUNCTION FOR
 THIS ENGLISH PEG SOLITARE PROBLEM WITH 8x8 BOARD.
 WHY?
 BECAUSE IT FINDS THE OPTIMAL SOLUTION BY VISITING 38 NODES ONLY.
 CONSIDERING THE FACT THAT BEST SOLUTION'S DEPTH WOULD BE 33, WE JUST LOOK AT 5 MORE NODES ONLY:)
 WE COULD HAVE ACHIVED 36 OR 35 BY INCREASING THE TUPLE LENGTH BUT THESE ARE FUNNY NUMBERS
 WHEN CONSIDERING GPU POWER.
 */
vector<int> heuristic(vector<vector<bool>>&board){
    int mansum = 0;
    int costsum = 0;
    int numofpegs = 0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(board[i][j]==1&&slots[i][j]!=-1){
                costsum += costpagoda[i][j]; // Pagoda cost of peg at i,j.
                mansum += abs(i-3)+abs(j-3); // Manhattan distance from center.
                numofpegs++; // Increment number of pegs by one.
            }
        }
    }
    countcalls++; // Increment the visited count.
    // We do this because that way the nodes earlier visited are picked first.
    // It is totally optional but we think that the nodes we go first would be
    // good option of all other heuristics are same.
    return {costsum,mansum,numofpegs,countcalls}; // Return the tuples.
}

// The main idea of the function same as all of the others. We just use priority queue
// as frontier thats all.
void DFS_Heuristic(double minutes){
    auto start = chrono::steady_clock::now();
    auto stop = chrono::steady_clock::now() + chrono::seconds((int)(60.0*minutes));
    
    priority_queue<pair<vector<int>,shared_ptr<Node>>,vector<pair<vector<int>,shared_ptr<Node>>>,greater<pair<vector<int>,shared_ptr<Node>>>>frontier;
    
    
    vector<vector<bool>> tempBoard = initializeBoard();
    frontier.push({heuristic(tempBoard),make_unique<Node>(tempBoard)});
    
    shared_ptr<Node> result = frontier.top().second;
    
    while(frontier.size()){
        auto curriter = frontier.top(); // Frontier is now a priority queue.
        frontier.pop();
        numberOfExpandedNodes++;
        
        shared_ptr<Node> curr = curriter.second;
        
        if(doWeStop(curr,stop)||printMemoryInMB(numberOfExpandedNodes)==-2){
            printResults(start,"Depth-First Search with a Node Selection Heuristic",minutes);
            return;
        }
        
        vector<vector<int>> removablePegs = getOrderedRemovablePegs(curr->board);
        vector<shared_ptr<Node>> childrenBoards = getBoardsAfterPegsRemoved(removablePegs, curr->board, curr);
        
        if(childrenBoards.empty()){ // Is this a sub-optimal solution?
            updateSubOptimalSolution(curr); // Keep the best sub-optimal solution.
        }
        
        for(auto child:childrenBoards){
            frontier.push({heuristic(child->board),child}); // Compute heuristic value and push.
            maxNumberOfStoredNodes = max((int)frontier.size(),maxNumberOfStoredNodes);
        }
    }
}




int main(int argc, char **argv)
{
    cout << "[INFO]: The program has started running." << endl << endl;
    //DFS_Standard(0.1);
    //BFS(0.1);
    //IDS(0.1);
    //DFS_RandomSelection(0.1);
    //DFS_Heuristic(0.1);
    return 0;
}
