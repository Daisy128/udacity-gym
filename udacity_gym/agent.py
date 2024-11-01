import pathlib
from tensorflow.keras.models import load_model
import numpy as np
import os
from .extras.model.lane_keeping.chauffeur.chauffeur_model import Chauffeur
from .extras.model.lane_keeping.dave.dave_model import Dave2
# import pygame
import torch
import torchvision
import tf2onnx
from onnx2pytorch import ConvertModel

from .action import UdacityAction
from .extras.model.lane_keeping.epoch.epoch_model import Epoch
from .extras.model.lane_keeping.vit.vit_model import ViT
from .observation import UdacityObservation
from utils.utils import preprocess

class UdacityAgent:

    def __init__(self, before_action_callbacks=None, after_action_callbacks=None, transform_callbacks=None):
        self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
        self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []
        self.transform_callbacks = transform_callbacks if transform_callbacks is not None else []

    def on_before_action(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.before_action_callbacks:
            callback(observation, *args, **kwargs)

    def on_after_action(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.after_action_callbacks:
            callback(observation, *args, **kwargs)

    def on_transform_observation(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.transform_callbacks:
            observation = callback(observation, *args, **kwargs)
        return observation

    def action(self, observation: UdacityObservation, *args, **kwargs):
        raise NotImplementedError('UdacityAgent does not implement __call__')

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        if observation.input_image is None:
            return UdacityAction(steering_angle=0.0, throttle=0.0)
        self.on_before_action(observation)
        observation = self.on_transform_observation(observation)
        action = self.action(observation, *args, **kwargs)
        self.on_after_action(observation, action=action)
        return action


class PIDUdacityAgent(UdacityAgent):

    def __init__(self, kp, kd, ki, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.ki = ki  # Integral gain
        self.prev_error = 0.0 # Calculate the rate of change of the error
        self.total_error = 0.0 # Sum of accumulated errors

        self.curr_sector = 0 # sector identifier
        self.skip_frame = 4 # skip observations under certain conditions: to stabilize control
        self.curr_skip_frame = 0

    def action(self, observation: UdacityObservation, *args, **kwargs):

        if observation.sector != self.curr_sector:
            if self.curr_skip_frame < self.skip_frame:
                self.curr_skip_frame += 1
            else:
                self.curr_skip_frame = 0
                self.curr_sector = observation.sector
            error = observation.cte
        else:
            error = (observation.next_cte + observation.cte) / 2
        diff_err = error - self.prev_error

        # Calculate steering angle
        steering_angle = - (self.kp * error) - (self.kd * diff_err) - (self.ki * self.total_error)
        steering_angle = max(-1, min(steering_angle, 1))

        # Calculate throttle
        throttle = 1

        # Save error for next prediction
        self.total_error += error
        self.total_error = self.total_error * 0.99
        self.prev_error = error

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)

class EndToEndLaneKeepingAgent(UdacityAgent):

    def __init__(self, model_name, checkpoint_path, before_action_callbacks=None, after_action_callbacks=None,
                 transform_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks, transform_callbacks)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        if model_name == "dave2":
            self.model = Dave2.load_from_checkpoint(self.checkpoint_path)
        if model_name == "epoch":
            self.model = Epoch.load_from_checkpoint(self.checkpoint_path)
        if model_name == "chauffeur":
            self.model = Chauffeur.load_from_checkpoint(self.checkpoint_path)
        if model_name == "vit":
            self.model = ViT.load_from_checkpoint(self.checkpoint_path)

    def action(self, observation: UdacityObservation, *args, **kwargs):

        # Cast input to right shape
        input_image = torchvision.transforms.ToTensor()(observation.input_image).to(self.model.device)

        # Calculate steering angle
        steering_angle = self.model(input_image).item()
        # Calculate throttle
        throttle = 0.22 - 0.5 * abs(steering_angle)

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)


class DaveUdacityAgent(UdacityAgent):

    def __init__(self, checkpoint_path, before_action_callbacks=None, after_action_callbacks=None,
                 transform_callbacks=None, model_type="PyTorch"):
        super().__init__(before_action_callbacks, after_action_callbacks, transform_callbacks)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.model_type = model_type
        if self.model_type=="PyTorch":
            self.model = Dave2.load_from_checkpoint(
                self.checkpoint_path, 
                map_location=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            )
        elif self.model_type=="Tensorflow":
            loaded_model = load_model(self.checkpoint_path)
            onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)
            self.model = ConvertModel(onnx_model)
        else:
            raise ValueError("checkpoint type must be 'PyTorch' or 'Tensorflow'.")


    def action(self, observation: UdacityObservation, *args, **kwargs):

        if self.model_type == "PyTorch":
            # Cast input to the right shape for PyTorch from [0, 255] to [0, 1]
            input_image = torchvision.transforms.ToTensor()(observation.input_image).to(self.model.device)
            
            # Calculate steering angle using the PyTorch model
            steering_angle = self.model(input_image).item()

        elif self.model_type == "Tensorflow":
            # Cast input to the right shape for TensorFlow
            input_image = np.expand_dims(observation.input_image, axis=0).astype('float32') / 255.0  # Add batch dimension and normalize
            
            # Calculate steering angle using the TensorFlow model
            steering_angle = self.model.predict(input_image)[0][0]

        # Calculate throttle
        throttle = 0.25 - 0.5 * abs(steering_angle)

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)

class SupervisedAgent(UdacityAgent):

    def __init__(
            self,
            model_path: str,
            max_speed: int,
            min_speed: int,
            predict_throttle: bool = False,
    ):
        super().__init__(before_action_callbacks=None, after_action_callbacks=None)

        # assert检查模型路径是否存在，不存在抛出错误信息
        assert os.path.exists(model_path), 'Model path {} not found'.format(model_path)

        self.model = load_model(model_path)
        self.predict_throttle = predict_throttle
        self.max_speed = max_speed
        self.min_speed = min_speed

    def action(self, observation: UdacityObservation, *args, **kwargs) -> UdacityAction:
        # observation by getting coordinate each time
        obs = observation.input_image # batch of images

        #print("Observations:", obs)
        obs = preprocess(obs)
        
        #  the model expects 4D array
        obs = np.array([obs])

        # obs = torch.transforms.Normalize(obs_mean,obs_std)
        speed = observation.speed
    
        if self.predict_throttle:
            action = self.model.predict(obs, batch_size=1, verbose=0)
            steering, throttle = action[0][0], action[0][1]
        else:
            import time
            time_start = time.time()
            steering = float(self.model.predict(obs, batch_size=1, verbose=0)[0])
            #print("DNN elasped time ",time.time() - time_start)
            steering = np.clip(steering, -1, 1)
            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed
            
            #steering = self.change_steering(steering=steering)
            #steering = float(self.model.predict(obs, batch_size=1, verbose=0))

            throttle = np.clip(a=1.0 - steering ** 2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)

            #print(f"steering {steering} throttle {throttle}")
            #self.model.summary()

        return UdacityAction(steering_angle=steering, throttle=throttle)
