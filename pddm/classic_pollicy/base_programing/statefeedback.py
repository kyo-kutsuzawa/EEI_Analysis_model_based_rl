#####################################
# Codes from:
#
# https://shizenkarasuzon.hatenablog.com/entry/2018/08/27/002812
#
# Modified by Chai Jiazheng
# E-mail: chai.jiazheng.q1@dc.tohoku.ac.jp
#
# 01/07/2019
#
#######################################
PI=3.14159265359

class STATEFEEDBACK:
    def __init__(self, Pth=5, Dth=1.5,Px=5,Dx=1.5,delta_time=0.04,target_pos=0,target_x=0):
        self.Kpth = Pth
        self.Kdth = Dth
        self.Kpx = Px
        self.Kdx = Dx
        self.targetPos = target_pos
        self.targetX = target_x
        self.delta_time=delta_time #Smallest timestep of the simulation
        self.clear()
        #print(delta_time)

    def clear(self):
        self.PthTerm = 0
        self.DthTerm = 0
        self.PxTerm = 0
        self.DxTerm = 0
        self.last_error_th = 0
        self.last_error_x = 0


        # Windup Guard
        self.windup_guard = 20.0
        self.output = 0.0

    def update(self, feedback_value_th,feedback_value_x):

        # Feedback value takes the target position as its reference.
        # For example, if target position is 0, then when the pole
        # tilts towards the rightside of the z-axis, its value is positive.
        # Else, its value is negative.
        if feedback_value_th>PI:
            feedback_value_th=-(PI-feedback_value_th%PI)

        error_th = self.targetPos - feedback_value_th
        error_x = self.targetPos - feedback_value_x

        delta_error_th = error_th - self.last_error_th
        delta_error_x = error_x - self.last_error_x

        self.PthTerm = self.Kpth * error_th
        self.PxTerm = self.Kpx * error_x



        self.DthTerm = delta_error_th / self.delta_time
        self.DxTerm = delta_error_x / self.delta_time
        self.last_error_th = error_th
        self.last_error_x = error_x
        self.output = self.PthTerm +  (self.Kdth * self.DthTerm) + self.PxTerm +  (self.Kdx * self.DxTerm)

        #if self.output >0.2:
         #   self.output = 0.2
        #elif self.output <-0.2:
           # self.output = -0.2


        return  self.output,feedback_value_th,feedback_value_x

    def setTargetPosition(self, targetPos):
        self.targetPos = targetPos