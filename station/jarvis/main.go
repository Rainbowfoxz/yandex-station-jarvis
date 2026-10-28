// jarvis — Джарвис без ПК: работает прямо на Яндекс Станции.
//
// jarvis-kws ловит «Джарвис» в микрофоне -> Алиса слушает вопрос (как на своё имя) -> её распознанный
// текст перехватываем, её ответ отменяем -> вопрос в нейросеть -> ответ читает Алиса своим голосом,
// команду («АЛИСА: включи музыку») выполняет она же.
//
//	jarvis -config /data/opt/jarvis/station/jarvis.conf
//
// В режиме pc (jarvis.conf: {"mode": "pc"}) нейросеть и голос — на ПК: вопрос уходит в канал /tmp/jarvis-q,
// его читает jarvis_wake.py --station.
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	_ "golang.org/x/crypto/x509roots/fallback" // своих корневых сертификатов на колонке может не быть
)

const (
	aliceAddr   = "127.0.0.1:9875"
	aliceIdle   = 60 * time.Second // соединение после простоя переоткрываем (см. jarvis_wake.py, 28.10)
	captureMax  = 12 * time.Second // дольше распознавания Алисы не ждём
	cooldown    = 2500 * time.Millisecond
	thinkMS     = 15000
	busyPhrase  = "Сэр, нейросеть сейчас недоступна, передаю вопрос Алисе."
	anim        = "/data/opt/think/anim.sh"
	thinkFrames = "/data/opt/think/thinking.bin"
	// Вопросы для ПК: его слушатель держит канал открытым на чтение. Никто не читает — открыть его на запись
	// без ожидания нельзя (ENXIO), и вопрос уходит Алисе.
	questionFIFO = "/tmp/jarvis-q"
	// Ловец имени: поток микрофона (pull.sh) -> jarvis-kws. Своя glibc и libvosk для aarch64 лежат в lib/:
	// 32-битная libvosk на прошивке Яндекса падает при загрузке (там libstdc++ без версий символов).
	defaultKWS = `sh /data/opt/jarvis/pull.sh | /data/opt/jarvis/station/lib/ld-linux-aarch64.so.1 ` +
		`--library-path /data/opt/jarvis/station/lib /data/opt/jarvis/station/jarvis-kws /data/opt/jarvis/station/model`
	beep         = `cd /system/vendor/quasar && LD_LIBRARY_PATH=/system/vendor/quasar:libs ./maind --gst-launch ` +
		`"filesrc location=/data/opt/jarvis/og/beep.wav ! decodebin ! audioconvert ! audioresample ! alsasink" ` +
		`>/dev/null 2>&1`
)

type Config struct {
	// pc — колонка только ловит «Джарвис» и перехватывает вопрос, а нейросеть и голос — на ПК
	// (jarvis_wake.py --station читает вопросы из questionFIFO); пусто — всё на колонке.
	Mode            string   `json:"mode"`
	Provider        string   `json:"provider"` // gemini | openai
	APIKey          string   `json:"api_key"`
	BaseURL         string   `json:"base_url"` // для openai: https://api.deepseek.com и т. п.
	Models          []string `json:"models"`
	MinimalThinking []string `json:"minimal_thinking"` // gemini: модели, принимающие thinkingLevel MINIMAL
	KWS             string   `json:"kws"`              // команда, печатающая WAKE на каждое «Джарвис»; пусто — defaultKWS
}

func logf(format string, args ...any) {
	fmt.Printf("%s %s\n", time.Now().Format("15:04:05"), fmt.Sprintf(format, args...))
}

// Alice — канал команд Алисе.
type Alice struct {
	mu   sync.Mutex
	conn net.Conn
	used time.Time
}

func (a *Alice) Send(frame []byte) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for attempt := 0; attempt < 2; attempt++ {
		if a.conn == nil || attempt > 0 || time.Since(a.used) > aliceIdle {
			if a.conn != nil {
				a.conn.Close()
			}
			conn, err := net.DialTimeout("tcp", aliceAddr, 5*time.Second)
			if err != nil {
				a.conn = nil
				logf("  нет связи с Алисой: %v", err)
				continue
			}
			a.conn = conn
		}
		a.conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
		if _, err := a.conn.Write(frame); err != nil {
			if attempt > 0 {
				logf("  нет связи с Алисой: %v", err)
			}
			continue
		}
		break
	}
	a.used = time.Now()
}

type Jarvis struct {
	alice *Alice
	brain *Brain // nil в режиме pc

	mu        sync.Mutex
	capturing time.Time // когда Алиса начала слушать вопрос для нас; ноль — не слушает
	busy      bool
	quietTill time.Time
}

// wake — «Джарвис» в микрофоне: Алиса приглушает музыку и слушает вопрос.
func (j *Jarvis) wake() {
	j.mu.Lock()
	defer j.mu.Unlock()
	now := time.Now()
	if j.busy || now.Before(j.quietTill) || (!j.capturing.IsZero() && now.Sub(j.capturing) < captureMax) {
		return
	}
	logf("«Джарвис» -> Алиса слушает вопрос")
	j.capturing = now
	j.quietTill = now.Add(cooldown)
	// Сигнал «слушаю»: имя засчитывается после паузы, и без сигнала вопрос начинали раньше, чем
	// Алиса включалась («…на планете» вместо «какая самая большая планета»).
	go exec.Command("sh", "-c", beep).Run() // Run, а не Start: иначе от каждого сигнала остаётся зомби
	j.alice.Send(listenFrame())
}

// recognized — Алиса распознала фразу. Наша, если мы её просили слушать.
func (j *Jarvis) recognized(text string) {
	j.mu.Lock()
	ours := !j.capturing.IsZero() && time.Since(j.capturing) < captureMax
	j.capturing = time.Time{}
	if !ours {
		j.mu.Unlock()
		return
	}
	// Сразу: ответ Алисы приходит примерно через секунду после распознавания.
	j.alice.Send(append(methodFrame(cancelDialogAndQueue), methodFrame(stopTTS)...))
	question := afterName(text)
	if question == "" || j.busy {
		j.mu.Unlock()
		logf("  Алиса распознала «%s» — вопроса нет", text)
		return
	}
	if j.brain == nil {
		j.mu.Unlock()
		j.toPC(question)
		return
	}
	j.busy = true
	j.mu.Unlock()
	go j.answer(question)
}

func (j *Jarvis) answer(question string) {
	defer func() {
		j.mu.Lock()
		j.busy = false
		j.quietTill = time.Now().Add(cooldown)
		j.mu.Unlock()
	}()
	logf("  вопрос: «%s»", question)
	stop := think()
	started := time.Now()
	kind, text, err := j.brain.Ask(question)
	stop()
	spent := time.Since(started).Seconds()
	switch {
	case err != nil:
		logf("  нейросеть недоступна (%.1f с) -> Алисе", spent)
		j.alice.Send(vocalizeFrame(busyPhrase))
		time.Sleep(speechTime(busyPhrase))
		j.alice.Send(textFrame(question))
	case kind == "alice":
		logf("  %s (%.1f с) -> Алисе: «%s»", j.brain.model, spent, text)
		j.alice.Send(textFrame(text))
	default:
		logf("  %s (%.1f с): %s", j.brain.model, spent, text)
		j.alice.Send(vocalizeFrame(text))
		time.Sleep(speechTime(text)) // своё «Джарвис» в ответе не должно будить нас же
	}
}

// toPC — вопрос слушателю на ПК; его нет — отвечает Алиса.
func (j *Jarvis) toPC(question string) {
	line := strings.ReplaceAll(question, "\n", " ") + "\n"
	fifo, err := os.OpenFile(questionFIFO, os.O_WRONLY|syscall.O_NONBLOCK, 0)
	if err == nil {
		_, err = fifo.WriteString(line)
		fifo.Close()
	}
	if err != nil {
		logf("  вопрос «%s»: ПК не слушает (%v) -> Алисе", question, err)
		j.alice.Send(textFrame(question))
		return
	}
	logf("  вопрос «%s» -> ПК", question)
}

// speechTime — сколько примерно Алиса читает текст (~14 символов в секунду).
func speechTime(text string) time.Duration {
	return time.Duration(len([]rune(text)))*70*time.Millisecond + time.Second
}

// think — анимация «думаю» на подсветке; возвращает функцию, которая её гасит.
func think() func() {
	cmd := exec.Command("sh", anim, thinkFrames, fmt.Sprint(thinkMS))
	if cmd.Start() != nil {
		return func() {}
	}
	return func() {
		cmd.Process.Kill()
		cmd.Wait()
		if fb, err := os.OpenFile("/dev/fb0", os.O_WRONLY, 0); err == nil {
			fb.Write(make([]byte, 356))
			fb.Close()
		}
	}
}

var nameRe = regexp.MustCompile(`^(?:джарвис|джарвиз|джервис|жарвис|джарвес|jarvis)[\s,.!?-]*`)

// afterName — фраза без обращения в начале: Алиса могла услышать и само «Джарвис».
func afterName(text string) string {
	t := strings.TrimSpace(strings.ToLower(text))
	return strings.TrimSpace(nameRe.ReplaceAllString(t, ""))
}

// events — постоянная подписка на события Алисы; ловит конец распознавания.
func (j *Jarvis) events() {
	for {
		conn, err := net.DialTimeout("tcp", aliceAddr, 5*time.Second)
		if err != nil {
			logf("нет связи с Алисой (%v), повтор через 3 с", err)
			time.Sleep(3 * time.Second)
			continue
		}
		conn.Write(subscribeFrame())
		r := bufio.NewReaderSize(conn, 1<<16)
		heard := ""
		for {
			var size uint32
			if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
				break
			}
			if size == 0 || size > 512*1024 {
				break
			}
			frame := make([]byte, size)
			if _, err := io.ReadFull(r, frame); err != nil {
				break
			}
			method, text, ok := aliceEvent(frame)
			if !ok {
				continue
			}
			if text != "" {
				heard = text
			}
			if method == recognitionCompleted {
				j.recognized(heard)
				heard = ""
			}
		}
		conn.Close()
		logf("канал событий Алисы закрылся, переподключаюсь")
		time.Sleep(time.Second)
	}
}

// listen — запускает ловца имени и перезапускает его, если он упал.
func (j *Jarvis) listen(command string) {
	for {
		cmd := exec.Command("sh", "-c", command)
		out, err := cmd.StdoutPipe()
		if err == nil {
			cmd.Stderr = os.Stderr
			err = cmd.Start()
		}
		if err != nil {
			logf("ловец имени не запустился: %v", err)
			time.Sleep(5 * time.Second)
			continue
		}
		lines := bufio.NewScanner(out)
		for lines.Scan() {
			switch lines.Text() {
			case "READY":
				logf("Джарвис слушает. Скажите «Джарвис».")
			case "WAKE":
				j.wake()
			}
		}
		cmd.Wait()
		logf("ловец имени завершился, перезапуск через 3 с")
		time.Sleep(3 * time.Second)
	}
}

func main() {
	path := flag.String("config", "/data/opt/jarvis/station/jarvis.conf", "файл настроек")
	ask := flag.String("ask", "", "проверка: задать вопрос нейросети и выйти (без колонки)")
	flag.Parse()
	raw, err := os.ReadFile(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "нет файла настроек:", err)
		os.Exit(1)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		fmt.Fprintln(os.Stderr, "файл настроек не JSON:", err)
		os.Exit(1)
	}
	if cfg.KWS == "" {
		cfg.KWS = defaultKWS
	}
	if cfg.Mode == "pc" {
		if err := syscall.Mkfifo(questionFIFO, 0600); err != nil && !os.IsExist(err) {
			fmt.Fprintln(os.Stderr, "не создать канал вопросов:", err)
			os.Exit(1)
		}
		j := &Jarvis{alice: &Alice{}}
		go j.events()
		j.listen(cfg.KWS)
		return
	}
	if cfg.APIKey == "" || len(cfg.Models) == 0 || (cfg.Provider != "gemini" && cfg.BaseURL == "") {
		fmt.Fprintln(os.Stderr, "в настройках нужны api_key, models, а для provider=openai ещё base_url")
		os.Exit(1)
	}
	brain := NewBrain(cfg)
	if *ask != "" {
		started := time.Now()
		kind, text, err := brain.Ask(*ask)
		if err != nil {
			fmt.Println("ошибка:", err)
			os.Exit(1)
		}
		fmt.Printf("%s (%s, %.1f с): %s\n", kind, brain.model, time.Since(started).Seconds(), text)
		return
	}
	j := &Jarvis{alice: &Alice{}, brain: brain}
	go j.events()
	j.listen(cfg.KWS)
}
