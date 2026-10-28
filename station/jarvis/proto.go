package main

// Кадры канала Алисы (порт 9875 на колонке) — тот же протокол, что в alice_say.py и ask_text.py:
// 4 байта длины (little-endian) + protobuf-конверт. Номера полей и методов — из quasar-protos
// (yandex_io/protos/remoting.proto).

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
)

const (
	startRequest         = 0  // AliceCapabilityMethod
	startConversation    = 2  // Алиса начинает слушать, как на своё имя
	cancelDialogAndQueue = 6
	stopTTS              = 14
	vocalize             = 18
	textInput            = 1 // VinsRequest.RequestType
	recognitionCompleted = 7 // AliceCapabilityListenerMethod
)

func varint(v uint64) []byte {
	return binary.AppendUvarint(nil, v)
}

func intField(field int, v uint64) []byte {
	return append(varint(uint64(field)*8), varint(v)...)
}

func bytesField(field int, v []byte) []byte {
	out := varint(uint64(field)*8 + 2)
	out = append(out, varint(uint64(len(v)))...)
	return append(out, v...)
}

func rpcFrame(field int, payload []byte) []byte {
	remote := append(bytesField(7, []byte("AliceCapability")), bytesField(field, payload)...)
	envelope := append(bytesField(252, remote), intField(253, 210)...)
	frame := binary.LittleEndian.AppendUint32(nil, uint32(len(envelope)))
	return append(frame, envelope...)
}

func subscribeFrame() []byte {
	return rpcFrame(20, bytesField(1, []byte("AliceCapability")))
}

// methodFrame — метод без параметров; silent=1 (поле 7): отмена без звука ошибки.
func methodFrame(method uint64) []byte {
	return rpcFrame(8, append(intField(1, method), intField(7, 1)...))
}

func listenFrame() []byte {
	return rpcFrame(8, intField(1, startConversation))
}

// vocalizeFrame — Алиса читает текст своим голосом (START_VOCALIZE, events=true, Append=1).
func vocalizeFrame(text string) []byte {
	directive := append(bytesField(1, []byte(text)), intField(2, 1)...)
	body := append(intField(1, vocalize), intField(3, 1)...)
	return rpcFrame(8, append(body, bytesField(13, directive)...))
}

// textFrame — вопрос или команда Алисе текстом, как из приложения.
func textFrame(text string) []byte {
	event, _ := json.Marshal(map[string]string{"type": "text_input", "text": text})
	vins := bytesField(1, []byte(uuid4()))
	vins = append(vins, bytesField(2, event)...)
	vins = append(vins, intField(4, 0)...)          // is_silent=0
	vins = append(vins, intField(12, 1)...)         // voice_session=1
	vins = append(vins, intField(18, textInput)...) // тип запроса
	body := append(intField(1, startRequest), bytesField(2, vins)...)
	return rpcFrame(8, append(body, intField(3, 1)...))
}

func uuid4() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

// fields разбирает protobuf-сообщение в «номер поля -> последнее значение»
// (uint64 для чисел, []byte для строк и вложенных сообщений).
func fields(data []byte) (map[int]any, error) {
	out := map[int]any{}
	for off := 0; off < len(data); {
		key, n := binary.Uvarint(data[off:])
		if n <= 0 || key>>3 == 0 {
			return nil, errors.New("битый protobuf")
		}
		off += n
		switch key & 7 {
		case 0:
			v, n := binary.Uvarint(data[off:])
			if n <= 0 {
				return nil, errors.New("битое число")
			}
			off += n
			out[int(key>>3)] = v
		case 1, 5:
			size := 8
			if key&7 == 5 {
				size = 4
			}
			if off+size > len(data) {
				return nil, errors.New("неполное поле")
			}
			off += size
		case 2:
			size, n := binary.Uvarint(data[off:])
			if n <= 0 || uint64(len(data)-off-n) < size {
				return nil, errors.New("неполное поле")
			}
			off += n
			out[int(key>>3)] = data[off : off+int(size)]
			off += int(size)
		default:
			return nil, errors.New("неизвестный тип поля")
		}
	}
	return out, nil
}

func sub(m map[int]any, field int) map[int]any {
	b, _ := m[field].([]byte)
	parsed, err := fields(b)
	if err != nil {
		return map[int]any{}
	}
	return parsed
}

// aliceEvent — событие слушателя Алисы: номер и распознанный текст (alice_event в jarvis_wake.py).
func aliceEvent(frame []byte) (method uint64, heard string, ok bool) {
	envelope, err := fields(frame)
	if err != nil {
		return 0, "", false
	}
	listener := sub(sub(envelope, 252), 18)
	if len(listener) == 0 {
		return 0, "", false
	}
	method, _ = listener[1].(uint64)
	text, _ := sub(sub(listener, 2), 15)[1].([]byte)
	return method, string(text), true
}
