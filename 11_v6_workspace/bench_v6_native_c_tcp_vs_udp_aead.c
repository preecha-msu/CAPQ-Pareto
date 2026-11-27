#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAX_CHUNK_PT 1360
#define HDR_LEN 12
#define TAG_LEN 16
#define MAX_FRAGS 128

static const uint8_t SESSION_KEY[32] = {
    0x42,0x19,0xa7,0x3c,0x88,0x05,0xe1,0xd4,0x71,0x2b,0x90,0xf3,0x66,0x18,0x5c,0xaa,
    0x31,0x7e,0xc9,0x04,0xb2,0x55,0x8d,0xe0,0x14,0x6a,0x9f,0x23,0xd8,0x47,0xbc,0x01
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int aead_seal(uint32_t req_id, uint16_t frag_id, uint16_t n_frags, uint32_t dir_flag,
                     const uint8_t *pt, int pt_len, uint8_t *out_pkt) {
    uint8_t iv[12];
    memcpy(iv, &dir_flag, 4);
    memcpy(iv + 4, &req_id, 4);
    memcpy(iv + 8, &frag_id, 2);
    memcpy(iv + 10, &n_frags, 2);

    uint32_t net_req = htonl(req_id);
    uint16_t net_fid = htons(frag_id);
    uint16_t net_nfr = htons(n_frags);
    uint32_t net_len = htonl((uint32_t)pt_len);
    memcpy(out_pkt, &net_req, 4);
    memcpy(out_pkt + 4, &net_fid, 2);
    memcpy(out_pkt + 6, &net_nfr, 2);
    memcpy(out_pkt + 8, &net_len, 4);

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len = 0, ct_len = 0;
    EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, SESSION_KEY, iv);
    EVP_EncryptUpdate(ctx, NULL, &len, out_pkt, HDR_LEN);
    EVP_EncryptUpdate(ctx, out_pkt + HDR_LEN, &len, pt, pt_len);
    ct_len = len;
    EVP_EncryptFinal_ex(ctx, out_pkt + HDR_LEN + len, &len);
    ct_len += len;
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, TAG_LEN, out_pkt + HDR_LEN + ct_len);
    EVP_CIPHER_CTX_free(ctx);
    return HDR_LEN + ct_len + TAG_LEN;
}

static int aead_open(uint32_t dir_flag, const uint8_t *pkt, int pkt_len,
                     uint32_t *req_id, uint16_t *frag_id, uint16_t *n_frags, uint8_t *out_pt) {
    if (pkt_len < HDR_LEN + TAG_LEN) return -1;
    uint32_t net_req, net_len;
    uint16_t net_fid, net_nfr;
    memcpy(&net_req, pkt, 4);
    memcpy(&net_fid, pkt + 4, 2);
    memcpy(&net_nfr, pkt + 6, 2);
    memcpy(&net_len, pkt + 8, 4);
    *req_id = ntohl(net_req);
    *frag_id = ntohs(net_fid);
    *n_frags = ntohs(net_nfr);
    int pt_len = (int)ntohl(net_len);
    if (pt_len < 0 || pt_len > MAX_CHUNK_PT || HDR_LEN + pt_len + TAG_LEN != pkt_len) return -2;

    uint8_t iv[12];
    memcpy(iv, &dir_flag, 4);
    memcpy(iv + 4, req_id, 4);
    memcpy(iv + 8, frag_id, 2);
    memcpy(iv + 10, n_frags, 2);

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len = 0, out_len = 0;
    EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, SESSION_KEY, iv);
    EVP_DecryptUpdate(ctx, NULL, &len, pkt, HDR_LEN);
    EVP_DecryptUpdate(ctx, out_pt, &len, pkt + HDR_LEN, pt_len);
    out_len = len;
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, TAG_LEN, (void *)(pkt + HDR_LEN + pt_len));
    int ok = EVP_DecryptFinal_ex(ctx, out_pt + len, &len);
    EVP_CIPHER_CTX_free(ctx);
    return (ok == 1) ? (out_len + len) : -3;
}

static int send_all(int fd, const uint8_t *buf, int len) {
    int sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, buf + sent, (size_t)(len - sent), MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        sent += (int)n;
    }
    return sent;
}

static int recv_all(int fd, uint8_t *buf, int len) {
    int got = 0;
    while (got < len) {
        ssize_t n = recv(fd, buf + got, (size_t)(len - got), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        got += (int)n;
    }
    return got;
}

/* Server mode: runs both TCP pooled AEAD server (:21443) and UDP paced AEAD+NACK server (:21444) */
static int udp_policing_mode = 0;

static void *tcp_srv_thread(void *arg) {
    (void)arg;
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(21443);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(lfd, (struct sockaddr *)&addr, sizeof(addr));
    listen(lfd, 16);

    uint8_t pkt[MAX_CHUNK_PT + 64], pt[MAX_CHUNK_PT], resp_src[MAX_CHUNK_PT];
    memset(resp_src, 0xA5, sizeof(resp_src));

    while (1) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        while (1) {
            if (recv_all(cfd, pkt, HDR_LEN) != HDR_LEN) break;
            uint32_t net_len;
            memcpy(&net_len, pkt + 8, 4);
            int pt_len = (int)ntohl(net_len);
            if (pt_len < 0 || pt_len > MAX_CHUNK_PT) break;
            if (recv_all(cfd, pkt + HDR_LEN, pt_len + TAG_LEN) != pt_len + TAG_LEN) break;
            uint32_t req_id = 0; uint16_t fid = 0, nfr = 0;
            if (aead_open(1U, pkt, HDR_LEN + pt_len + TAG_LEN, &req_id, &fid, &nfr, pt) < 0) break;

            if (fid + 1 == nfr) {
                uint32_t net_resp_bytes;
                memcpy(&net_resp_bytes, pt, 4);
                int resp_total = (int)ntohl(net_resp_bytes);
                int resp_frags = (resp_total + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
                for (int f = 0; f < resp_frags; f++) {
                    int rem = resp_total - f * MAX_CHUNK_PT;
                    int clen = rem > MAX_CHUNK_PT ? MAX_CHUNK_PT : rem;
                    int slen = aead_seal(req_id, (uint16_t)f, (uint16_t)resp_frags, 2U, resp_src, clen, pkt);
                    if (send_all(cfd, pkt, slen) != slen) break;
                }
            }
        }
        close(cfd);
    }
    return NULL;
}

static int run_server(int policed_flag) {
    udp_policing_mode = policed_flag;
    pthread_t tid;
    pthread_create(&tid, NULL, tcp_srv_thread, NULL);
    pthread_detach(tid);

    int ufd = socket(AF_INET, SOCK_DGRAM, 0);
    int buf_sz = 8 * 1024 * 1024;
    setsockopt(ufd, SOL_SOCKET, SO_RCVBUF, &buf_sz, sizeof(buf_sz));
    setsockopt(ufd, SOL_SOCKET, SO_SNDBUF, &buf_sz, sizeof(buf_sz));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(21444);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(ufd, (struct sockaddr *)&addr, sizeof(addr));

    uint8_t pkt[2048], pt[MAX_CHUNK_PT], out_pkt[2048], resp_src[MAX_CHUNK_PT];
    memset(resp_src, 0x5C, sizeof(resp_src));
    uint32_t cur_req = 0xFFFFFFFFU;
    int cur_resp_total = 1024;
    int got_cnt = 0;
    uint8_t got_map[MAX_FRAGS];

    while (1) {
        struct sockaddr_in cli;
        socklen_t clen = sizeof(cli);
        ssize_t n = recvfrom(ufd, pkt, sizeof(pkt), 0, (struct sockaddr *)&cli, &clen);
        if (n <= 0) continue;

        /* Control NACK frame: magic 0x4E41434B ('NACK') + req_id(4B) + missing_count(2B) + list of uint16_t */
        if (n >= 10) {
            uint32_t magic;
            memcpy(&magic, pkt, 4);
            if (ntohl(magic) == 0x4E41434BU) {
                uint32_t r_id; uint16_t m_cnt;
                memcpy(&r_id, pkt + 4, 4);
                memcpy(&m_cnt, pkt + 8, 2);
                r_id = ntohl(r_id); m_cnt = ntohs(m_cnt);
                int resp_frags = (cur_resp_total + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
                for (int i = 0; i < m_cnt && 10 + 2 * i + 2 <= n; i++) {
                    uint16_t mf;
                    memcpy(&mf, pkt + 10 + 2 * i, 2);
                    mf = ntohs(mf);
                    if (mf < resp_frags) {
                        if (udp_policing_mode && (mf % 2 == 0)) continue;
                        int rem = cur_resp_total - mf * MAX_CHUNK_PT;
                        int ch = rem > MAX_CHUNK_PT ? MAX_CHUNK_PT : rem;
                        int sl = aead_seal(r_id, mf, (uint16_t)resp_frags, 2U, resp_src, ch, out_pkt);
                        sendto(ufd, out_pkt, (size_t)sl, 0, (struct sockaddr *)&cli, clen);
                    }
                }
                continue;
            }
        }

        uint32_t req_id = 0; uint16_t fid = 0, nfr = 0;
        int plen = aead_open(1U, pkt, (int)n, &req_id, &fid, &nfr, pt);
        if (plen < 0 || nfr > MAX_FRAGS) continue;

        /* Enterprise middlebox RFC 9308 UDP rate-policing simulation on application bursts (>160 B) */
        if (udp_policing_mode && plen > 160) {
            usleep(185000); /* Token-bucket policing queue delay + drop */
            if ((req_id + fid) % 3 != 0) continue;
        }

        if (req_id != cur_req) {
            cur_req = req_id;
            memset(got_map, 0, sizeof(got_map));
            got_cnt = 0;
            if (plen >= 4) {
                uint32_t nrb;
                memcpy(&nrb, pt, 4);
                cur_resp_total = (int)ntohl(nrb);
            }
        }
        if (!got_map[fid]) {
            got_map[fid] = 1;
            got_cnt++;
        }
        if (got_cnt == nfr) {
            int resp_frags = (cur_resp_total + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
            for (int f = 0; f < resp_frags; f++) {
                int rem = cur_resp_total - f * MAX_CHUNK_PT;
                int ch = rem > MAX_CHUNK_PT ? MAX_CHUNK_PT : rem;
                int sl = aead_seal(req_id, (uint16_t)f, (uint16_t)resp_frags, 2U, resp_src, ch, out_pkt);
                sendto(ufd, out_pkt, (size_t)sl, 0, (struct sockaddr *)&cli, clen);
                if ((f + 1) % 16 == 0) usleep(120); /* Lightweight user-space burst pacing */
            }
        }
    }
    return 0;
}

static double client_tcp_rpc(int tfd, uint32_t req_id, int req_bytes, int resp_bytes) {
    uint8_t pt[MAX_CHUNK_PT], pkt[MAX_CHUNK_PT + 64];
    memset(pt, 0x33, sizeof(pt));
    uint32_t nrb = htonl((uint32_t)resp_bytes);
    memcpy(pt, &nrb, 4);

    double t0 = now_ms();
    int req_frags = (req_bytes + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
    for (int f = 0; f < req_frags; f++) {
        int rem = req_bytes - f * MAX_CHUNK_PT;
        int ch = rem > MAX_CHUNK_PT ? MAX_CHUNK_PT : rem;
        if (ch < 4) ch = 4;
        int sl = aead_seal(req_id, (uint16_t)f, (uint16_t)req_frags, 1U, pt, ch, pkt);
        if (send_all(tfd, pkt, sl) != sl) return 950.0;
    }
    int resp_frags = (resp_bytes + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
    for (int f = 0; f < resp_frags; f++) {
        if (recv_all(tfd, pkt, HDR_LEN) != HDR_LEN) return 950.0;
        uint32_t net_len;
        memcpy(&net_len, pkt + 8, 4);
        int plen = (int)ntohl(net_len);
        if (recv_all(tfd, pkt + HDR_LEN, plen + TAG_LEN) != plen + TAG_LEN) return 950.0;
        uint32_t rid = 0; uint16_t fid = 0, nfr = 0;
        if (aead_open(2U, pkt, HDR_LEN + plen + TAG_LEN, &rid, &fid, &nfr, pt) < 0) return 950.0;
    }
    return now_ms() - t0;
}

static double client_udp_rpc(int ufd, const struct sockaddr_in *srv, uint32_t req_id,
                             int req_bytes, int resp_bytes, double pto_ms) {
    uint8_t pt[MAX_CHUNK_PT], pkt[2048];
    memset(pt, 0x44, sizeof(pt));
    uint32_t nrb = htonl((uint32_t)resp_bytes);
    memcpy(pt, &nrb, 4);

    int req_frags = (req_bytes + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
    int resp_frags = (resp_bytes + MAX_CHUNK_PT - 1) / MAX_CHUNK_PT;
    uint8_t rx_map[MAX_FRAGS] = {0};
    int rx_cnt = 0;

    double t0 = now_ms();
    for (int attempt = 0; attempt < 6 && rx_cnt < resp_frags; attempt++) {
        if (attempt == 0 || rx_cnt == 0) {
            for (int f = 0; f < req_frags; f++) {
                int rem = req_bytes - f * MAX_CHUNK_PT;
                int ch = rem > MAX_CHUNK_PT ? MAX_CHUNK_PT : rem;
                if (ch < 4) ch = 4;
                int sl = aead_seal(req_id, (uint16_t)f, (uint16_t)req_frags, 1U, pt, ch, pkt);
                sendto(ufd, pkt, (size_t)sl, 0, (const struct sockaddr *)srv, sizeof(*srv));
            }
        } else {
            /* Send selective NACK for missing fragments */
            uint32_t magic = htonl(0x4E41434BU);
            uint32_t n_rid = htonl(req_id);
            uint16_t miss = 0;
            memcpy(pkt, &magic, 4);
            memcpy(pkt + 4, &n_rid, 4);
            for (int f = 0; f < resp_frags; f++) {
                if (!rx_map[f]) {
                    uint16_t nf = htons((uint16_t)f);
                    memcpy(pkt + 10 + 2 * miss, &nf, 2);
                    miss++;
                }
            }
            uint16_t n_miss = htons(miss);
            memcpy(pkt + 8, &n_miss, 2);
            sendto(ufd, pkt, (size_t)(10 + 2 * miss), 0, (const struct sockaddr *)srv, sizeof(*srv));
        }

        double deadline = now_ms() + pto_ms;
        while (rx_cnt < resp_frags) {
            double rem_ms = deadline - now_ms();
            if (rem_ms <= 0.2 || (now_ms() - t0) >= 950.0) break;
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(ufd, &rfds);
            struct timeval tv;
            tv.tv_sec = (int)(rem_ms / 1000.0);
            tv.tv_usec = (int)((rem_ms - tv.tv_sec * 1000.0) * 1000.0);
            int rv = select(ufd + 1, &rfds, NULL, NULL, &tv);
            if (rv <= 0) break;
            ssize_t n = recv(ufd, pkt, sizeof(pkt), 0);
            if (n <= 0) continue;
            uint32_t rid = 0; uint16_t fid = 0, nfr = 0;
            if (aead_open(2U, pkt, (int)n, &rid, &fid, &nfr, pt) >= 0 && rid == req_id && fid < resp_frags) {
                if (!rx_map[fid]) {
                    rx_map[fid] = 1;
                    rx_cnt++;
                }
            }
        }
        if ((now_ms() - t0) >= 950.0) return 950.0;
    }
    return (rx_cnt == resp_frags) ? (now_ms() - t0) : 600.0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "server") == 0) {
        int pol = (argc >= 3) ? atoi(argv[2]) : 0;
        return run_server(pol);
    }
    /* Client mode: bench <srv_ip> <regime_name> <rtt_ms> <n_pairs> */
    const char *srv_ip = argv[1];
    const char *regime = argv[2];
    double rtt_ms = atof(argv[3]);
    int n_pairs = atoi(argv[4]);
    double pto_ms = (rtt_ms * 1.55 + 6.0 < 18.0) ? 18.0 : (rtt_ms * 1.55 + 6.0);

    int tfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(tfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    struct sockaddr_in taddr = {0};
    taddr.sin_family = AF_INET;
    taddr.sin_port = htons(21443);
    inet_pton(AF_INET, srv_ip, &taddr.sin_addr);
    if (connect(tfd, (struct sockaddr *)&taddr, sizeof(taddr)) != 0) {
        perror("tcp connect");
        return 1;
    }

    int ufd = socket(AF_INET, SOCK_DGRAM, 0);
    int bsz = 8 * 1024 * 1024;
    setsockopt(ufd, SOL_SOCKET, SO_RCVBUF, &bsz, sizeof(bsz));
    setsockopt(ufd, SOL_SOCKET, SO_SNDBUF, &bsz, sizeof(bsz));
    struct sockaddr_in uaddr = {0};
    uaddr.sin_family = AF_INET;
    uaddr.sin_port = htons(21444);
    inet_pton(AF_INET, srv_ip, &uaddr.sin_addr);

    struct { const char *name; int req_b; int resp_b; } workloads[3] = {
        {"W1_SQL_320B_512B",         320,    512},
        {"W2_Vec_INT8_210B_1024B",   210,   1024},
        {"W3_Conf_1CT_129KiB",      1024, 132106}
    };

    uint32_t req_seq = 100;
    /* Warmup */
    for (int w = 0; w < 3; w++) {
        (void)client_tcp_rpc(tfd, req_seq++, 210, 512);
        (void)client_udp_rpc(ufd, &uaddr, req_seq++, 210, 512, pto_ms);
    }

    for (int w = 0; w < 3; w++) {
        int rqb = workloads[w].req_b;
        int rsb = workloads[w].resp_b;
        int rfc9308_tcp_hold = 0;
        int consec_udp_stalls = 0;

        for (int i = 0; i < n_pairs; i++) {
            double lat_tcp = 0.0, lat_udp = 0.0, lat_rule = 0.0;
            /* ABBA counterbalanced order */
            if (i % 2 == 0) {
                lat_tcp = client_tcp_rpc(tfd, req_seq++, rqb, rsb);
                lat_udp = client_udp_rpc(ufd, &uaddr, req_seq++, rqb, rsb, pto_ms);
            } else {
                lat_udp = client_udp_rpc(ufd, &uaddr, req_seq++, rqb, rsb, pto_ms);
                lat_tcp = client_tcp_rpc(tfd, req_seq++, rqb, rsb);
            }

            /* Stateless Native-C RFC 9308 Timeout-to-TCP Fallback Rule */
            if (rfc9308_tcp_hold > 0) {
                lat_rule = client_tcp_rpc(tfd, req_seq++, rqb, rsb);
                rfc9308_tcp_hold--;
            } else {
                lat_rule = client_udp_rpc(ufd, &uaddr, req_seq++, rqb, rsb, pto_ms);
                if (lat_rule >= 250.0) {
                    consec_udp_stalls++;
                    if (consec_udp_stalls >= 2 || lat_rule >= 550.0) {
                        rfc9308_tcp_hold = 32;
                        consec_udp_stalls = 0;
                    }
                } else {
                    consec_udp_stalls = 0;
                }
            }
            printf("RAW,%s,%s,%d,%.4f,%.4f,%.4f\n",
                   regime, workloads[w].name, i, lat_tcp, lat_udp, lat_rule);
        }
    }
    close(tfd);
    close(ufd);
    return 0;
}
