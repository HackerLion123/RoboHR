from django.shortcuts import render,render_to_response
from django.http import HttpResponse, Http404, HttpResponseRedirect
from .process import get_barchart,get_graph
from django.core.files.base import ContentFile
from binascii import a2b_base64
from models.roboHR import read_image
from .models import *
import os
import base64


def video_stream(request):
	name,emotion = read_image(url)


def add_user(request):
	if request.method == 'POST':
		user = User()
		name = request.POST['username']
		email = request.POST['email']
		if request.POST['password'] == request.POST['again_pass']:
			password = request.POST['password']
			user = User(user_name=name,email=email,password=password,profile_pic=None,first_login=True)
			if request.FILES:
				user.profile_pic = request.FILES['pro_pic']
			user.save()
			return render(request,"html/signin.html",{'title':"Login/Signup",'message':"account created successfully"})
		else:
			return render(request,"html/signin.html",{'title':"Login/Signup",'message':"password don't match"}) 

def index(request):
	if request.session.get('user_id'):
		return render(request,"index.html",{'title':"Home | Welcome",'user':request.session['user_id'],'logged_in':True})
	else:
		return render(request,"index.html",{'title':"Home | Welcome",'user':"RoboHR",'logged_in':False})

def model(request):
	if request.session.get('user_id'):
		return render(request,"html/model.html",{'title':"Model",'logged_in':True,'user':request.session['user_id']})
	else:
		return render(request,"html/model.html",{'title':"Model",'logged_in':False})

def bot(request):
	if request.session.get('user_id'):
		return render(request,"html/ruby.html",{'title':"Chatbot",'logged_in':True,'user':request.session['user_id']})
	else:
		return render(request,"html/ruby.html",{'title':"Chatbot",'logged_in':False})

def signup(request):
	if not request.session.get('user_id'):
		return render(request,"html/signin.html",{'title':"Login/Signup"})
	else:
		return HttpResponseRedirect('/')

def dashboard(request):
	path = ""
	if request.session.get('user_id'):
		name = os.path.join(path,"{}.csv".format(request.session['user_id']))
	else:
		name = os.path.join(path,"unknown.csv")
	try:	
		bar = get_barchart(name)
		graph = get_graph(name)
		return render(request,"html/main.html",{'title':"Dashboard",'graph':graph,'bar':bar,'file':True})
	except Exception as e:
		return render(request,"html/main.html",{'title':"Dashboard",'file':False})
def lazyloading(request):
	return render()

def about(request):
	if request.session.get('user_id'):
		return render(request,"html/about.html",{'title':"About"})
	else:
		return render(request,"html/about.html",{'title':"About"})

def login(request):
	try:
		m = User.objects.get(user_name=request.POST['user'])
		if m.password == request.POST['password']:
			request.session['user_id'] = m.user_name
			return HttpResponseRedirect("/")
		else:
			return HttpResponse("your username and password didn't match")
	except User.DoesNotExist:
		return HttpResponse("not registered")

def logout(request):
	try:
		del request.session['user_id']
		return HttpResponseRedirect("/")
	except KeyError:
		pass

def upload(request):
	try:
		if request.method == 'POST':
			print(request.POST['base64'])
			url = request.POST['base64']
			img = handlebase64(url)
			name,emotion = read_image(img)
			return render(request,"html/model.html",{'title':model,'src':url,'name':name,'emotion':emotion})
	except Exception as e:
		return render(request,"html/model.html",{'title':model,'src':"",'name':"can't detect face",'emotion':"hello"})

def handlebase64(file):
	encoding = file.split(',')[-1]

	return encoding.encode()