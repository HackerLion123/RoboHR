from django.db import models


class ImageModel(models.Model):

	model_pic = models.ImageField(upload_to = "captures/", default = None)
	timestamp = models.DateTimeField(auto_now_add=True)

class User(models.Model):
	
	user_name = models.CharField(max_length=128,primary_key=True)	
	email = models.CharField(max_length=128)
	password = models.CharField(max_length=128)
	first_login = models.BooleanField(default=True)
	profile_pic = models.ImageField(upload_to = "captures/",null=True, default = None)

	def __str__(self):
		return self.user_name