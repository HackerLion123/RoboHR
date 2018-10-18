from django.conf.urls import url
from . import views
from django.contrib.auth import login,logout
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
	url(r'^$',views.index,name="home"),
	url(r'^cam/$',views.model,name="model"),
	url(r'^bot/$',views.bot,name="model"),
	url(r'^dashboard/$',views.dashboard,name="model"),
	url(r'^about/$',views.about,name="about"),
	url(r'^signup/$',views.signup,name="signup"),
	url(r'^login/$',views.login,name="login"),
	url(r'^logout/$',views.logout,name="logout"),
	url(r'^upload/$',views.upload,name="upload"),
	url(r'^adduser/$',views.add_user,name="adduser"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)