from django.shortcuts import render
from django.http import HttpResponse, Http404, JsonResponse

import cv2


def video_stream():
	pass

def index(request):
	return render(request,"index.html")

def model(request):
	return render(request,"html/model.html")

def bot(request):
	return render(request,"html/ruby.html")

def dashboard(request):
	return render(request,"html/main.html")

def lazyloading(request):
	return render()

def about(request):
	return render(request,"html/about.html")
