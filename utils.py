import string
from underthesea import word_tokenize
import re
from collections import deque
from tqdm import tqdm
from rank_bm25 import *
import numpy as np

number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
stop_word = number + chars + ["này", "về", "của", "các", "được","như", "là", "sau", "và", "do", "hoặc",  "dưới", "đây", 
                              "nếu", "có", "khác", "đến", "việc", "đó", "vào", "mà", "không", "sự", "trong", "theo", "tại", 
                              "khoản", "cho", "từ", "phải",  "ngày",  "để", "bộ",  "với", "năm", "khi", "số", "trên", "đã", 
                              "thì", "thuộc", "điểm", "đồng", "một", "bị", "lại", "ở", "làm", "nơi", "dễ", "”", "“"]

# replace by regex numbered and letter bullets from articles
regx_numbered_list = r"\n+\d{1,3}\."
regx_letter_list = r"\n+[abcdđefghijklmnopqrstuvxyz]\)"

def remove_stopword(w):
    return w not in stop_word
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    text = re.sub(regx_numbered_list, " ", text)
    text = re.sub(regx_letter_list, " ", text)
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

def bm25_rank_paragraphs(query, paragraphs, k1: float = 0.6, b: float = 0.6):
    documents = []
    for p in paragraphs:
        tokens = bm25_tokenizer(p)
        documents.append(tokens)
    
    bm25 = BM25Plus(documents, k1=k1, b=b)

    tokenized_query = bm25_tokenizer(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top1_idx = np.argpartition(doc_scores, -1)[-1:]
    
    return paragraphs[top1_idx[0]]

def sbert_rank_paragraphs(query, paragraphs, model, util):
    embeddings = []
    for p in paragraphs:
        embedded = model.encode(p)
        embeddings.append(embedded)
    np_embeddings = np.array(embeddings)

    encoded_query = model.encode(query)

    all_cosine = util.cos_sim(encoded_query, np_embeddings).numpy().squeeze(0)
    top1_idx = np.argpartition(all_cosine, -1)[-1:]

    return paragraphs[top1_idx[0]]

def aggregate_embeddings(paragraphs, model):
    embeddings = []
    for p in paragraphs:
        embedded = model.encode(p)
        embeddings.append(embedded)
    np_embeddings = np.array(embeddings) 

    return np.average(np_embeddings, axis=0)

def segment_long_text(long_text: str, max_length: int = 2000, sliding_thresh: int = 1000):
    """
    This function use dynamic sliding window over a list of passages with each passage is split by new-line character.
    The output of this function is a list of paragraphs with overlapping text. The step size (number of passages shifted)
    of the sliding window so that the total number of characters of shifted passages lower than `sliding threshold`.
    varies depending on 
    - long_text: a long text to be split into passages with length shorter than `max_length`
    - max_length: maximum length in character for each paragraph
    - output: a list of paragraphs derives from the input long text
    """
    paragraphs = []

    passages = re.split(r'\n+', long_text)
    lengths = [len(passage) for passage in passages]


    # print (passages)
    # print (lengths)
    # print (sum(lengths))

    queue = deque([])
    total_length = 0
    shift = 0
    for idx, length in tqdm(enumerate(lengths)):
        # print ("idx: ", idx)
        if total_length > 0:
            if total_length + length < max_length:
                queue.append(length)
                total_length += length
            elif total_length + length == max_length:
                    queue.append(length)
                    total_length = max_length
                    joint_passages = ' '.join(passages[idx_2] for idx_2 in range(idx - len(queue) + 1, idx + 1))
                    paragraphs.append(joint_passages)
                    # print (">> total_length + length = max_length")
                    # print ("total_length: ", total_length)
                    # print ("queue: ", queue)

                    # print (idx, [ind for ind in range(idx - len(queue) + 1, idx + 1)])

                    shift = 0
                    while len(queue) > 0 and shift + queue[0] < sliding_thresh:
                        shift += queue[0]
                        total_length -= queue.popleft()
            elif total_length + length > max_length:
                joint_passages = ' '.join(passages[idx_2] for idx_2 in range(idx - len(queue), idx))
                paragraphs.append(joint_passages)
                # print (">> total_length + length > max_length")
                # print ("total_length: ", total_length)
                # print ("queue: ", queue)

                # print (idx, [ind for ind in range(idx - len(queue), idx)])
              
                if length >= max_length:
                    paragraphs.append(passages[idx])
                    # print (">> total_length + length > max_length & length >= max_length")
                    # print ("total_length: ", total_length)
                    # print ("queue: ", queue)
                    # print (idx, [idx])
                    queue.clear()
                    total_length = 0
                else:
                    queue.append(length)
                    total_length += length
                    
                    shift = 0
                    while len(queue) > 0 and shift + queue[0] < sliding_thresh:
                        shift += queue[0]
                        total_length -= queue.popleft()                            
                    
                    # shift as long as total_length < max_length
                    while total_length >= max_length:
                        total_length -= queue.popleft()
        else:
            if length < max_length:
                queue.append(length)
                total_length += length
            else:
                paragraphs.append(passages[idx])
                # print (">> total_length = 0 & length < max_length")
                # print ("total_length: ", total_length)
                # print ("queue: ", queue)
                # print (idx, [idx])
    if total_length > 0:
        joint_passages = ' '.join(passages[idx_2] for idx_2 in range(len(lengths) - len(queue), len(lengths)))
        paragraphs.append(joint_passages)

    return paragraphs

if __name__ == "__main__":

    query = "Thời hạn thẩm định báo cáo đánh giá tác động môi trường là bao nhiêu ngày?"
    
    long_text = "Thẩm định báo cáo đánh giá tác động môi trường\n\n1. Hồ sơ đề nghị thẩm định báo cáo đánh giá tác động môi trường bao gồm:\n\na) Văn bản đề nghị thẩm định báo cáo đánh giá tác động môi trường;\n\nb) Báo cáo đánh giá tác động môi trường;\n\nc) Báo cáo nghiên cứu khả thi hoặc tài liệu tương đương với báo cáo nghiên cứu khả thi của dự án đầu tư.\n\n2. Đối với dự án đầu tư xây dựng thuộc đối tượng phải được cơ quan chuyên môn về xây dựng thẩm định báo cáo nghiên cứu khả thi theo quy định của pháp luật về xây dựng, chủ dự án đầu tư được trình đồng thời hồ sơ đề nghị thẩm định báo cáo đánh giá tác động môi trường với hồ sơ đề nghị thẩm định báo cáo nghiên cứu khả thi; thời điểm trình do chủ dự án đầu tư quyết định nhưng phải bảo đảm trước khi có kết luận thẩm định báo cáo nghiên cứu khả thi.\n\n3. Việc thẩm định báo cáo đánh giá tác động môi trường được quy định như sau:\n\na) Cơ quan thẩm định ban hành quyết định thành lập hội đồng thẩm định gồm ít nhất là 07 thành viên; gửi quyết định thành lập hội đồng kèm theo tài liệu quy định tại điểm b và điểm c khoản 1 Điều này đến từng thành viên hội đồng;\n\nb) Hội đồng thẩm định phải có ít nhất một phần ba tổng số thành viên là chuyên gia. Chuyên gia là thành viên hội đồng phải có chuyên môn về môi trường hoặc lĩnh vực khác có liên quan đến dự án đầu tư và có kinh nghiệm công tác ít nhất là 07 năm nếu có bằng cử nhân hoặc văn bằng trình độ tương đương, ít nhất là 03 năm nếu có bằng thạc sĩ hoặc văn bằng trình độ tương đương, ít nhất là 02 năm nếu có bằng tiến sĩ hoặc văn bằng trình độ tương đương;\n\nc) Chuyên gia tham gia thực hiện đánh giá tác động môi trường của dự án đầu tư không được tham gia hội đồng thẩm định báo cáo đánh giá tác động môi trường của dự án đó;\n\nd) Trường hợp dự án đầu tư có hoạt động xả nước thải vào công trình thủy lợi thì hội đồng thẩm định phải có đại diện cơ quan nhà nước quản lý công trình thủy lợi đó; cơ quan thẩm định phải lấy ý kiến bằng văn bản và đạt được sự đồng thuận của cơ quan nhà nước quản lý công trình thủy lợi đó trước khi phê duyệt kết quả thẩm định.\n\nCơ quan nhà nước quản lý công trình thủy lợi có trách nhiệm cử thành viên tham gia hội đồng thẩm định, có ý kiến bằng văn bản về việc phê duyệt kết quả thẩm định trong thời hạn lấy ý kiến; trường hợp hết thời hạn lấy ý kiến mà không có văn bản trả lời thì được coi là đồng thuận với nội dung báo cáo đánh giá tác động môi trường;\n\nđ) Thành viên hội đồng thẩm định có trách nhiệm nghiên cứu hồ sơ đề nghị thẩm định, viết bản nhận xét về nội dung thẩm định quy định tại khoản 7 Điều này và chịu trách nhiệm trước pháp luật về ý kiến nhận xét, đánh giá của mình;\n\ne) Cơ quan thẩm định xem xét, đánh giá và tổng hợp ý kiến của các thành viên hội đồng thẩm định, ý kiến của cơ quan, tổ chức có liên quan (nếu có) để làm căn cứ quyết định việc phê duyệt kết quả thẩm định báo cáo đánh giá tác động môi trường.\n\n4. Trường hợp cần thiết, cơ quan thẩm định tổ chức khảo sát thực tế, lấy ý kiến của cơ quan, tổ chức và chuyên gia để thẩm định báo cáo đánh giá tác động môi trường.\n\n5. Trong thời gian thẩm định, trường hợp có yêu cầu chỉnh sửa, bổ sung báo cáo đánh giá tác động môi trường, cơ quan thẩm định có trách nhiệm thông báo bằng văn bản cho chủ dự án đầu tư để thực hiện.\n\n6. Thời hạn thẩm định báo cáo đánh giá tác động môi trường được tính từ ngày nhận được đầy đủ hồ sơ hợp lệ và được quy định như sau:\n\na) Không quá 45 ngày đối với dự án đầu tư nhóm I quy định tại khoản 3 Điều 28 của Luật này;\n\nb) Không quá 30 ngày đối với dự án đầu tư nhóm II quy định tại các điểm c, d, đ và e khoản 4 Điều 28 của Luật này;\n\nc) Trong thời hạn quy định tại điểm a và điểm b khoản này, cơ quan thẩm định có trách nhiệm thông báo bằng văn bản cho chủ dự án đầu tư về kết quả thẩm định. Thời gian chủ dự án đầu tư chỉnh sửa, bổ sung báo cáo đánh giá tác động môi trường theo yêu cầu của cơ quan thẩm định và thời gian xem xét, ra quyết định phê duyệt quy định tại khoản 9 Điều này không tính vào thời hạn thẩm định;\n\nd) Thời hạn thẩm định quy định tại điểm a và điểm b khoản này có thể được kéo dài theo quyết định của Thủ tướng Chính phủ.\n\n7. Nội dung thẩm định báo cáo đánh giá tác động môi trường bao gồm:\n\na) Sự phù hợp với Quy hoạch bảo vệ môi trường quốc gia, quy hoạch vùng, quy hoạch tỉnh, quy định của pháp luật về bảo vệ môi trường;\n\nb) Sự phù hợp của phương pháp đánh giá tác động môi trường và phương pháp khác được sử dụng (nếu có);\n\nc) Sự phù hợp về việc nhận dạng, xác định hạng mục công trình và hoạt động của dự án đầu tư có khả năng tác động xấu đến môi trường;\n\nd) Sự phù hợp của kết quả đánh giá hiện trạng môi trường, đa dạng sinh học; nhận dạng đối tượng bị tác động, yếu tố nhạy cảm về môi trường nơi thực hiện dự án đầu tư;\n\nđ) Sự phù hợp của kết quả nhận dạng, dự báo các tác động chính, chất thải phát sinh từ dự án đầu tư đến môi trường; dự báo sự cố môi trường;\n\ne) Sự phù hợp, tính khả thi của các công trình, biện pháp bảo vệ môi trường; phương án cải tạo, phục hồi môi trường (nếu có); phương án bồi hoàn đa dạng sinh học (nếu có); phương án phòng ngừa, ứng phó sự cố môi trường của dự án đầu tư;\n\ng) Sự phù hợp của chương trình quản lý và giám sát môi trường; tính đầy đủ, khả thi đối với các cam kết bảo vệ môi trường của chủ dự án đầu tư.\n\n8. Thủ tướng Chính phủ quyết định việc tổ chức thẩm định báo cáo đánh giá tác động môi trường của dự án đầu tư vượt quá khả năng thẩm định trong nước, cần thuê tư vấn nước ngoài thẩm định. Kết quả thẩm định báo cáo đánh giá tác động môi trường của tư vấn nước ngoài là cơ sở để cơ quan nhà nước có thẩm quyền quy định tại Điều 35 của Luật này phê duyệt kết quả thẩm định báo cáo đánh giá tác động môi trường.\n\n9. Trong thời hạn 20 ngày kể từ ngày nhận được báo cáo đánh giá tác động môi trường đã được chỉnh sửa, bổ sung theo yêu cầu (nếu có) của cơ quan thẩm định, người đứng đầu cơ quan thẩm định có trách nhiệm ra quyết định phê duyệt kết quả thẩm định báo cáo đánh giá tác động môi trường; trường hợp không phê duyệt thì phải trả lời bằng văn bản cho chủ dự án đầu tư và nêu rõ lý do.\n\n10. Việc gửi hồ sơ đề nghị thẩm định báo cáo đánh giá tác động môi trường, tiếp nhận, giải quyết và thông báo kết quả thẩm định báo cáo đánh giá tác động môi trường được thực hiện thông qua một trong các hình thức gửi trực tiếp, qua đường bưu điện hoặc bản điện tử thông qua hệ thống dịch vụ công trực tuyến theo đề nghị của chủ dự án đầu tư.\n\n11. Bộ trưởng Bộ Tài nguyên và Môi trường quy định chi tiết về tổ chức và hoạt động của hội đồng thẩm định; công khai danh sách hội đồng thẩm định; biểu mẫu văn bản, tài liệu của hồ sơ đề nghị thẩm định báo cáo đánh giá tác động môi trường, quyết định phê duyệt kết quả thẩm định báo cáo đánh giá tác động môi trường; thời hạn lấy ý kiến quy định tại điểm d khoản 3 Điều này."

    # query = "Trường hợp thông tin trong Sổ hộ khẩu còn hiệu lực có thông tin khác với thông tin trong Cơ sở dữ liệu về cư trú thì sử dụng thông tin trong Cơ sở dữ liệu về cư trú, đúng hay sai?"

    # long_text = "1. Việc điều chỉnh thông tin về cư trú của công dân được thực hiện trong các trường hợp sau đây:\na) Thay đổi chủ hộ;\nb) Thay đổi thông tin về hộ tịch so với thông tin đã được lưu trữ trong Cơ sở dữ liệu về cư trú;\nc) Thay đổi địa chỉ nơi cư trú trong Cơ sở dữ liệu về cư trú do có sự điều chỉnh về địa giới đơn vị hành chính, tên đơn vị hành chính, tên đường, phố, tổ dân phố, thôn, xóm, làng, ấp, bản, buôn, phum, sóc, cách đánh số nhà.\n2. Hồ sơ điều chỉnh thông tin về cư trú quy định tại điểm a và điểm b khoản 1 Điều này bao gồm:\na) Tờ khai thay đổi thông tin cư trú;\nb) Giấy tờ, tài liệu chứng minh việc điều chỉnh thông tin.\n3. Thủ tục điều chỉnh thông tin về cư trú được thực hiện như sau:\na) Đối với trường hợp quy định tại điểm a khoản 1 Điều này, thành viên hộ gia đình nộp hồ sơ quy định tại khoản 2 Điều này đến cơ quan đăng ký cư trú. Trong thời hạn 03 ngày làm việc kể từ ngày nhận được hồ sơ đầy đủ và hợp lệ, cơ quan đăng ký cư trú có trách nhiệm điều chỉnh thông tin về chủ hộ trong Cơ sở dữ liệu về cư trú và thông báo cho thành viên hộ gia đình về việc đã cập nhật thông tin; trường hợp từ chối điều chỉnh thì phải thông báo bằng văn bản và nêu rõ lý do;\nb) Đối với trường hợp quy định tại điểm b khoản 1 Điều này, trong thời hạn 30 ngày kể từ ngày có quyết định của cơ quan có thẩm quyền thay đổi thông tin về hộ tịch, người có thông tin được điều chỉnh nộp hồ sơ đăng ký điều chỉnh thông tin có liên quan trong Cơ sở dữ liệu về cư trú quy định tại khoản 2 Điều này đến cơ quan đăng ký cư trú.\nTrong thời hạn 03 ngày làm việc kể từ ngày nhận được hồ sơ đầy đủ và hợp lệ, cơ quan đăng ký cư trú có trách nhiệm điều chỉnh thông tin về hộ tịch trong Cơ sở dữ liệu về cư trú và thông báo cho người đăng ký về việc đã cập nhật thông tin; trường hợp từ chối điều chỉnh thì phải thông báo bằng văn bản và nêu rõ lý do;\nc) Đối với trường hợp quy định tại điểm c khoản 1 Điều này, cơ quan đăng ký cư trú có trách nhiệm điều chỉnh, cập nhật việc thay đổi thông tin trong Cơ sở dữ liệu về cư trú."


    paragraphs = segment_long_text(long_text)

    candidate = bm25_rank_paragraphs(query, paragraphs)

    print (candidate)