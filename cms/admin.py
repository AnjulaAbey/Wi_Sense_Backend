from django.contrib import admin
from .models import CSIData

class ListAdmin(admin.ModelAdmin):
    search_fields = ()

    def get_list_display(self, request):
        return [
            field.name 
            for field in self.model._meta.concrete_fields 
            if field.name != "id" and not field.many_to_many
        ]

# Register your models here.
admin.site.register(CSIData, ListAdmin)

