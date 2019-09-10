
import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
import math
##########################
ConstraintA=[[-1,1],
            [8,2],
            [-1,0],
            [0,-1]]
ConstraintB=[2,17,0,0]
N=2 #length of variable
objective_cost=[5.5,2.1]
class Node:
    id=None
    Parent_id=None
    direction=None #0 is left and 1 is right
    A=ConstraintA
    b=ConstraintB
    scan_status=0
    variable=[]
    optimal_value=0
    index_variable=0#this is use to indicate which variable is prunning

    status=None
    def sovling(self):
        x=cp.Variable(N)
        obj=np.array(objective_cost)*x

        #print(np.array(self.A).shape)
        constr=[(np.array(self.A)*x)<=np.array(self.b).transpose()]
        prob = cp.Problem(cp.Maximize(obj), constr)
        print(prob)
        prob.solve()
        self.variable=x.value
        self.optimal_value=prob.value
        print("optimal",self.optimal_value)
        #print(x.value)
        if prob.status not in ["infeasible", "unbounded"]:
            self.status=1
            return True
        else:
            self.status=0
            return False

    def addConstr(self,constr_A,constr_b):
        self.A.append(constr_A)
        self.b.append(constr_b)
current_node=Node()
current_node.id=0
listnode=[]
listnode.append(current_node)
#node.addConstr([1,0],1)
status=current_node.sovling()
current_node.status=status
#using the stack to add the node
stack=[]
if status==1:
    current_node.scan_status=1
    stack.append(current_node.id)

#constructing the tree
while 1:
    #check the left
    if current_node.status==1:
        #prunning
        #go to the left node
        index = current_node.index_variable
        left_node=Node()
        left_node.id=len(listnode)
        temp_constraint=np.zeros(N)
        temp_constraint[index]=1
        #print(index)
        print("variable",current_node.variable)
        temp_lowbound=math.floor(current_node.variable[index])

        # add constraint of the left node
        left_node.addConstr(temp_constraint,temp_lowbound)
        status=left_node.sovling()
        listnode.append(left_node)
        print("Solving left node:",status)
        if status==0:
            flag=0
        else:
            left_node.index_variable=index+1
            stack.append(left_node)

        #go to the right node
        right_node=Node()
        index = current_node.index_variable
        right_node = Node()
        right_node.id = len(listnode)
        temp_constraint = np.zeros(N)
        temp_constraint[index] = -1
        temp_upperbound = math.ceil(current_node.variable[index])
        # add constraint of the left node
        right_node.addConstr(temp_constraint, -temp_upperbound)
        status = left_node.sovling()
        listnode.append(right_node)
        print("Solving right node:", status)
        if status == 0:
            flag = 0
        else:
            right_node.index_variable = index + 1
            stack.append(right_node)
        if flag==0:
            Current_node=stack.pop()

    if len(stack)==0:
        break
#input,hidden,output, batch size
n_in, n_h, n_out,batch_size=2,5,1,5
cmax=14.08
input_data=[]
input_data.append([1.3,3.3])
output_data=[]
output_data.append([1])
x=1.3
y=3.3
#using flag to indicate integer value
flagx=0
flagy=0
while 0:
    if x-np.round(x,0)<=0.00001:
        flagx=1
    if x-np.round(y,0)<=0.00001:
        flagy=1


x=torch.tensor([[1.3,3.3]])
print(x)
y=torch.tensor([[1.0]])
model=nn.Sequential(nn.Linear(n_in,n_h),
                    nn.ReLU(),
                    nn.Linear(n_h,n_out),
                    nn.Sigmoid())
criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(1):
    #forward prop
    y_pred=model(x)
    #ompute and print loss
    loss=criterion(y_pred,y)
    print('epoch:',epoch,' loss',loss.item())
    #zero the gradients
    optimizer.zero_grad()
    #perform backward
    loss.backward()
    #update the parameters
    optimizer.step()
#
# Import packages.

