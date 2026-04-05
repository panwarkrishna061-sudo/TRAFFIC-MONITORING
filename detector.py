import cv2

custom_classes = ['ambulance', 'firetruck', 'rikshaw']
general_classes = ['car', 'bus', 'truck', 'motorcycle']
priority_classes = ['ambulance', 'firetruck']
CONF = 0.25

def detect_all(frame, custom_model=None, general_model=None):
    total_count = 0
    priority_count = 0
    normal_count = 0
    priority_detected = False

    # Custom model detection
    if custom_model is not None:
        results1 = custom_model(frame, conf=CONF, verbose=False)

        for r in results1:
            if r.boxes is None:
                continue

            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                name = custom_model.names[int(cls)]

                if name in custom_classes:
                    x1, y1, x2, y2 = map(int, box)

                    if name in priority_classes:
                        color = (0, 0, 255)  # Red
                        label = f"{name} PRIORITY {float(conf):.2f}"
                        priority_detected = True
                        priority_count += 1
                    else:
                        color = (0, 255, 0)  # Green
                        label = f"{name} {float(conf):.2f}"
                        normal_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                    )

                    total_count += 1

    # General model detection
    if general_model is not None:
        results2 = general_model(frame, conf=CONF, verbose=False)

        for r in results2:
            if r.boxes is None:
                continue

            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                name = general_model.names[int(cls)]

                if name in general_classes:
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"{name} {float(conf):.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
                    )

                    normal_count += 1
                    total_count += 1

    # Dashboard
    cv2.rectangle(frame, (10, 10), (420, 140), (0, 0, 0), -1)

    cv2.putText(frame, f"Total Count: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Priority Count: {priority_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Normal Count: {normal_count}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if priority_detected:
        cv2.putText(frame, "PRIORITY VEHICLE DETECTED!", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame