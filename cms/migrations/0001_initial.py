# Generated by Django 5.0.1 on 2025-01-09 05:33

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CSIData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(max_length=255)),
                ('mac', models.CharField(max_length=255)),
                ('rssi', models.FloatField()),
                ('time_stamp', models.DateTimeField()),
                ('raw_data', models.JSONField()),
            ],
        ),
    ]
