from numpy import linspace
from secrets import choice
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
from manipulator_TDC_sid_wrench import Manipulator
import csv

class ArmSim():
    def __init__(self):
        self.robot = Manipulator([0.65,0.3,0.77],[0.7,0.1,0.89])
        self.ee_pos_data = []
        self.timearray, self.t_step= np.linspace(0,25,2500,retstep=True)
        self.torque_data = np.empty([np.size(self.timearray),len(self.robot.rev)])
        self.alpha_in = [ 0 for i in range(len(self.robot.rev))]
        #print(self.t_step)
        self.h = 1/101
        self.torque_h = np.array([0,0,0,0,0,0])
        self.torque = np.array([0,0,0,0,0,0])        
        self.k_not = np.diag(np.array([1e0,1e0,1e0,1e0,1e0,1e0]))
        self.ydot = np.array([0,0,0,0,0,0])

        self.error = np.empty((2500,6))


    def runSim(self):
        self.robot.setInitialState()
        self.robot.turnOffActuators()
        self.robot.turnOffDamping()
        p.enableJointForceTorqueSensor(self.robot.arm,7,enableSensor=1)
        for i,t in zip(range(np.size(self.timearray)),self.timearray):
            p.stepSimulation()
            time.sleep(self.h)
            # print(self.get_sigma(t, self.k_not, self.alpha_in, self.robot.eef_vel, self.ydot))            
            self.sigma = self.get_sigma(self.h, self.k_not, self.alpha_in, self.robot.eef_vel, self.ydot)            
            self.torque_h = self.torque
            self.torque = self.get_torque(t,i)
            self.torque_data[i] = self.torque
            self.ee_pos_data.append(list(self.robot.forces))
            p.setJointMotorControlArray(self.robot.arm,self.robot.rev,controlMode=p.TORQUE_CONTROL,forces=self.torque)
            np.append(self.error, self.positionError())
            self.pose = np.array(list(p.getLinkState(self.robot.arm,8,computeLinkVelocity=True)[0])+list(p.getEulerFromQuaternion(p.getLinkState(self.robot.arm,8,computeLinkVelocity=True)[1]))) 

            p.addUserDebugLine([0,0,0], self.pose[:3], lineColorRGB = [1,1,0], lineWidth = 1, lifeTime = 0.099)

        # print(self.error)


        # fig, ax = plt.subplots()

        # transposed_data = self.error.T

        # # Create a figure and axis
        # fig, ax = plt.subplots()

        # # Plot each transposed row as a line
        # for row in transposed_data:
        #     ax.plot(row)
        # plt.show()

    def positionError(self):
        self.positionerror =  self.x_des - np.array(self.robot.eef_pos)
        # print(self.robot.eef_pos[3:])
        return self.positionerror
    
    def velocityError(self):
        self.velocityerror = self.xd_des - np.array(self.robot.eef_vel)
        return self.velocityerror

    def get_sigma(self, h, k_not, alpha_in, eff_vel, ydot):
        # yddot = alpha_in + k_not(eff_vel - ydot)
        ydot = ydot + h*(alpha_in + k_not@(eff_vel - ydot))
        return k_not@(eff_vel - ydot)

    def get_torque(self,t,i):
        self.robot.joint_state_info()
        self.robot.getjacobian()
        self.m = self.robot.massMatrix()
        # if i == 0:
        #     print(self.m)

        # q_ddot_h = (np.array(self.robot.angles) - 2*np.array(self.robot.angles_prev) + np.array(self.robot.angles_pprev))/(self.h)**2
        q_ddot_h = (np.array(self.robot.omega_prev) - np.array(self.robot.omega_pprev))/self.h
        # torque_h = self.torque_data[i - 1]
        # print(self.robot.angles)
        # print(self.robot.angles_prev)
        # print(self.robot.angles_pprev)
        

        # print(np.array(self.robot.angles),np.array(2*self.robot.angles_prev), np.array(self.robot.angles_pprev) )
        #Impedance Control
        self.M = np.diag(np.array([1e0,1e0,1e0,1e0,1e0,1e0]))
        self.Kp = np.diag([120,120,120,120,120,120])
        self.Kd = np.diag(6*[2*np.sqrt(180)])
        
        self.M_bar = np.array([ [ 9.25958840e-01 ,-3.44950225e-01 , 5.07159378e-02 ,-3.04338478e-04, 3.28488058e-04, -2.17260720e-04] ,
                                [-3.44950225e-01,  1.76554356e+00 , 3.04762975e-01 , 1.33003450e-02, 2.34989579e-07 , 8.67767088e-07],
                                [ 5.07159378e-02,  3.04762975e-01 , 6.10282299e-01 , 1.02895488e-02 ,2.34990129e-07,  8.67767088e-07],
                                [-3.04338478e-04 , 1.33003450e-02 , 1.02895488e-02 , 5.85027636e-03 ,2.34990939e-07 , 8.67767088e-07],
                                [ 3.28488058e-04  ,2.34989579e-07 , 2.34990129e-07 , 2.34990939e-07 ,2.08277673e-03 , 6.39887126e-25],
                                [-2.17260720e-04,  8.67767088e-07,  8.67767088e-07 , 8.67767088e-07, 6.39887126e-25 , 2.20016010e-04]])
        #self.x_des,self.xd_des,self.xdd_des = self.robot.getrunningTraj((t-100)/100)

        #Desired Reference
        #self.x_des = np.array([0.6+0.05*np.sin(t),0.4+0.05*np.cos(t),0.77,0.1,1,0.403])
        #self.xd_des = np.array([0.05*np.cos(t),-0.05*np.sin(t),0,0,0,0])s
        #self.xdd_des = np.array([-0.05*np.sin(t),-0.05*np.cos(t),0,0,0,0])
        
    
        self.x_des = np.array([0.65+0.08*np.sin(t),0.3+0.08*np.cos(t),0.57,0.1+0.1*np.sin(t),1.3,0.403])
        self.xd_des = np.array([0.08*np.cos(t),-0.08*np.sin(t),0,0.1*np.cos(t),0,0])
        self.xdd_des = np.array([-0.08*np.sin(t),-0.08*np.cos(t),0,-0.1*np.sin(t),0,0])
        # print("wrench", self.robot.wrench())
        # print("contact", self.robot.contactForce())
        self.alpha_in = self.xdd_des + np.linalg.inv(self.M)@(self.Kp@self.positionError() + self.Kd@self.velocityError()+self.robot.wrench())
        # print( "1",self.alpha_in)        
        self.alpha_in = self.alpha_in - self.sigma
        # print("2", self.alpha_in)
        # self.alpha_in = self.xdd_des + np.linalg.inv(self.M)@(self.Kp@self.positionError() + self.Kd@self.velocityError()+self.robot.contactForce())
        #self.alpha_in = self.xdd_des + np.linalg.inv(self.M)@(self.Kp@self.positionError() + self.Kd@self.velocityError())
        self.ah = np.linalg.inv(self.robot.analyticjacobian())@(self.alpha_in - self.robot.Jdot@self.robot.omega)
        # print(self.m@self.ah + self.robot.coriolisVector() + self.robot.gravityVector()-self.robot.analyticjacobian().T@self.robot.contactForce())
        # print(torque_h - self.M_bar@q_ddot_h + self.M_bar@self.ah )
        # return self.m@self.ah + self.robot.coriolisVector() + self.robot.gravityVector()-self.robot.analyticjacobian().T@self.robot.contactForce()
        #return self.m@self.ah + self.robot.coriolisVector() + self.robot.gravityVector()
        return self.torque_h - self.M_bar@q_ddot_h + self.M_bar@self.ah 

    #def write_data(self,t):
        #fields = ['ideal','actual']
        #filename = "plotting_data.csv"
        # writing to csv file
        #with open(filename, 'w') as csvfile:
            # creating a csv dict writer object
            #writer = csv.writer(csvfile)
            #writer.writerow(fields)
            #writer.writerows(self.robot.eef_pos)
    #def write_data(self):
     #timearray = linspace(0,4,10000)
     #plt.plot(self.t_step,self.robot.eef_pos[2])
     #plt.show()
    

    def update_line(self):

        hl.set_xdata(np.append(hl.get_xdata(), 1))
        hl.set_ydata(np.append(hl.get_ydata(), self.robot.eef_pos[0]))
        #print(hl.get_ydata())
        plt.draw()
        plt.show()
            

if __name__ == "__main__":
    hl, = plt.plot([], [])

    r1 = ArmSim()
    p1 = np.array([0.7,0.1,0.89,0.1,1.2, 0.203])
    p2 = np.array([0.7,0.1,0.77,0.1,1.2, 0.403])
    p3 = np.array([0.7,0.1,0.77,0.1,1.2, 0.203])
    p4 = np.array([0.7,0.1,0.77,0.1,1.2, 0.403])
    r1.robot.setTrajPt(p1,p2,p3,p4)
    r1.robot.get_B()
    # print(r1.get_torque())

    r1.runSim()
    # r1.update_line()

    # r1.move_on_table()
    


