from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /greyscaler/5/
    url(r'^out/$', views.im, name='im'),
    # ex: /greyscaler/5/
    url(r'^$', views.model_form_upload, name='index'),
]