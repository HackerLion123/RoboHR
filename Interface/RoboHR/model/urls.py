from django.conf.urls import url
from . import views


urlpatterns = {
	url(r'^$',views.index,name="home"),
	url(r'^cam/$',views.model,name="model"),
	url(r'^bot/$',views.bot,name="model"),
	url(r'^dashboard/$',views.dashboard,name="model"),
	url(r'^about/$',views.about,name="about"),
}