from django.http import HttpResponse
from django.template import loader
from . import forms
from .models import Document
from django.shortcuts import render, redirect
from .im_proc import *
import os


def im(request):
    template = loader.get_template('greyscaler/result.html')
    context = {}
    return HttpResponse(template.render(context, request))


def model_form_upload(request):
    if request.method == 'POST':
        form = forms.DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            fp = Document.objects.last()
            fp_im = str(fp.image)
            im = open_and_orient(fp_im)
            r, c = im.width, im.height
            while r > 1000:
                im = im.resize((int(r / 2), int(c / 2)))
                r, c = im.size
            while c > 1000:
                im = im.resize((int(r / 2), int(c / 2)))
                r, c = im.size
            im2 = np.array(im)  # need to copy im to release it from memory so it can be deleted
            del im
            os.remove(str(fp_im))
            grey = im2[:, :, 1].astype(np.uint8)
            pca_im = (pca(im2) * 255).astype(np.uint8)
            pca_im2 = (pca(im2, component=1) * 255).astype(np.uint8)
            avg_pca_cust = (avgLabPCAPCAwLabPCA_custom_coefs(im2) * 255).astype(np.uint8)
            avg_pca_Y = (avgYCrCbPCAPCAwLabPCA(im2) * 255).astype(np.uint8)
            Image.fromarray(pca_im).save('static/pca.jpg')
            Image.fromarray(pca_im2).save('static/pca2.jpg')
            Image.fromarray(avg_pca_cust).save('static/avg_pca_cust.jpg')
            Image.fromarray(avg_pca_Y).save('static/avg_pca_Y.jpg')
            Image.fromarray(grey).save('static/grey.jpg')
            Image.fromarray(im2).save('static/orig.jpg')
            return redirect('im')
    else:
        form = forms.DocumentForm()
    return render(request, 'greyscaler/upload.html', {
        'form': form
    })
