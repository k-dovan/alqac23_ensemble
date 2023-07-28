from transformers import pipeline
model_checkpoint = "nguyenvulebinh/vi-mrc-large"
# model_checkpoint = "saved_model/finetuned_vi_mrc_large"
nlp = pipeline('question-answering', model=model_checkpoint,
                   tokenizer=model_checkpoint)
QA_input = {
  'question': "Cơ quan nào có trách nhiệm thống nhất quản lý nhà nước về điện ảnh?",
  'context': "Trách nhiệm quản lý nhà nước về điện ảnh của Chính phủ, Bộ Văn hóa, Thể thao và Du lịch\n\n1. Chính phủ thống nhất quản lý nhà nước về điện ảnh.\n\n2. Bộ Văn hóa, Thể thao và Du lịch là cơ quan giúp Chính phủ thực hiện quản lý nhà nước về điện ảnh và có trách nhiệm sau đây:\n\na) Ban hành hoặc trình cơ quan nhà nước có thẩm quyền ban hành và tổ chức thực hiện chính sách, văn bản quy phạm pháp luật về điện ảnh, chiến lược, kế hoạch phát triển điện ảnh;\n\nb) Thông tin, tuyên truyền, phổ biến, giáo dục pháp luật về điện ảnh;\n\nc) Xây dựng tiêu chuẩn quốc gia, quy chuẩn kỹ thuật trong hoạt động điện ảnh; hệ thống chỉ tiêu thống kê, cơ sở dữ liệu ngành điện ảnh;\n\nd) Đào tạo, bồi dưỡng và phát triển nguồn nhân lực điện ảnh;\n\nđ) Hợp tác quốc tế trong hoạt động điện ảnh; quảng bá, xúc tiến phát triển điện ảnh trong nước và nước ngoài;\n\ne) Nghiên cứu, ứng dụng khoa học và công nghệ trong hoạt động điện ảnh;\n\ng) Cấp, thu hồi giấy phép trong hoạt động điện ảnh; dừng phổ biến phim theo thẩm quyền;\n\nh) Thực hiện công tác thi đua, khen thưởng trong hoạt động điện ảnh;\n\ni) Thanh tra, kiểm tra, giải quyết khiếu nại, tố cáo và xử lý vi phạm pháp luật trong hoạt động điện ảnh theo thẩm quyền."
}
res = nlp(QA_input)
print('pipeline: {}'.format(res))