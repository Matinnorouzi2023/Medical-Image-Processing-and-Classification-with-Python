# 🏥 Medical Image Processing & Classification Toolkit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

یک ابزار جامع برای پردازش و تحلیل تصاویر پزشکی با استفاده از یادگیری عمیق

## 📑 فهرست مطالب
- [مقدمه](#-مقدمه)
- [امکانات](#-امکانات)
- [نصب و راه‌اندازی](#-نصب-و-راه‌اندازی)
- [آموزش گام به گام](#-آموزش-گام-به-گام)
- [دیتاست‌های پشتیبانی شده](#-دیتاست‌های-پشتیبانی-شده)
- [نتایج نمونه](#-نتایج-نمونه)
- [مشارکت](#-مشارکت)
- [مجوز](#-مجوز)

## 🌟 مقدمه
این پروژه ابزاری کامل برای پردازش تصاویر پزشکی و طبقه‌بندی بیماری‌ها ارائه می‌دهد. با استفاده از این کتابخانه می‌توانید:

- تصاویر DICOM و NIfTI را پردازش کنید
- مدل‌های یادگیری عمیق برای تشخیص بیماری‌ها آموزش دهید
- تصاویر پزشکی را سگمنت کنید

## 🛠 امکانات
### پردازش تصاویر
- پیش‌پردازش تصاویر پزشکی
- نرمال‌سازی هیستوگرام
- حذف نویز

### مدل‌های از پیش آموزش دیده
- طبقه‌بندی پنومونی از تصاویر X-Ray
- سگمنتاسیون قلب در تصاویر MRI
- تشخیص تومورهای مغزی

## 🔧 نصب و راه‌اندازی
1. ابتدا محیط مجازی ایجاد کنید:
```bash
python -m venv venv
source venv/bin/activate  # برای لینوکس/مک
venv\Scripts\activate  # برای ویندوز
