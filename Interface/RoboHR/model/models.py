from django.db import models

# Create your models here.

class ImageModel(models.Model):

	model_pic = models.ImageField(upload_to = "captures/", default = None)