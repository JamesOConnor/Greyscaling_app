from django.db import models


class Document(models.Model):
    image = models.FileField(upload_to='static/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.description
