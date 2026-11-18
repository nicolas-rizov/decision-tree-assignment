import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. Load Data
df = pd.read_csv('clinic_appointments.csv')
df['target'] = (df['no_show'] == 'no-show').astype(int)  # 1 = no-show, 0 = checked-in

features = [
    'lead_time_days', 'prior_no_shows', 'age', 'distance_miles',
    'sms_reminder', 'new_patient', 'appointment_hour'
]
X = df[features]
y = df['target']

# 2. Stratified 70/30 Train/Test Split (preserves ~22.8% no-show rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print(f"Total rows: {len(df)} | Train: {len(X_train)} ({y_train.mean():.1%} no-show) | Test: {len(X_test)} ({y_test.mean():.1%} no-show)")

# 3. Fit Decision Tree (v1: max_depth=3, min_samples_leaf=5)
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

print("\n--- Tree Decision Rules ---")
print(export_text(clf, feature_names=features))

# Save clean tree visualization for your slide
plt.figure(figsize=(14, 7))
plot_tree(
    clf,
    feature_names=features,
    class_names=['checked-in (Hold)', 'no-show (Overbook)'],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=False
)
plt.title("Clinic No-Show Decision Tree (v1, max_depth=3)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300)
plt.close()
print("Saved 'decision_tree.png'")

# 4. Evaluate Baseline vs. v1 on Test Set (120 appointments)
def evaluate_policy(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    total_val = tn * 0 + fp * (-60) + fn * (-120) + tp * 0
    per_appt = total_val / len(y_true)
    print(f"\n--- {name} (Test Set, n={len(y_true)}) ---")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy: {acc:.1%} | Precision: {prec:.1%} | Recall: {rec:.1%}")
    print(f"Total Test Value: ${total_val:,.2f} (${per_appt:,.2f} per appointment)")
    return total_val, per_appt

# Baseline: Predict show (0) for all appointments
y_pred_baseline = np.zeros_like(y_test)
base_total, base_per_appt = evaluate_policy(y_test, y_pred_baseline, "Baseline (Hold All)")

# v1 Model
train_acc = accuracy_score(y_train, clf.predict(X_train))
y_pred_v1 = clf.predict(X_test)
v1_total, v1_per_appt = evaluate_policy(y_test, y_pred_v1, "v1 Decision Tree")
print(f"v1 Train Accuracy: {train_acc:.1%} vs. Test Accuracy: {accuracy_score(y_test, y_pred_v1):.1%}")

# Annualized Value (Assuming 10,000 appointments/year: ~40/day * 250 days)
annual_volume = 10000
annual_savings = (v1_per_appt - base_per_appt) * annual_volume
print(f"\nNet Savings per Appointment: +${v1_per_appt - base_per_appt:,.2f}")
print(f"Annualized Value Improvement (at {annual_volume:,} appts/yr): +${annual_savings:,.2f}/yr")

# 5. Decision Function & Upcoming Appointments D and E
def decide_slot(appointment_dict):
    row_df = pd.DataFrame([appointment_dict])[features]
    pred = clf.predict(row_df)[0]
    prob_no_show = clf.predict_proba(row_df)[0][1]
    leaf_id = clf.apply(row_df)[0]
    decision = "OVERBOOK" if pred == 1 else "HOLD"
    return decision, leaf_id, prob_no_show

appt_D = {
    'lead_time_days': 21, 'prior_no_shows': 1, 'age': 27,
    'distance_miles': 9.5, 'sms_reminder': 0, 'new_patient': 1, 'appointment_hour': 15
}
appt_E = {
    'lead_time_days': 3, 'prior_no_shows': 0, 'age': 61,
    'distance_miles': 2.0, 'sms_reminder': 1, 'new_patient': 0, 'appointment_hour': 9
}

for label, appt in [("D", appt_D), ("E", appt_E)]:
    decision, leaf, prob = decide_slot(appt)
    print(f"Appointment {label}: Leaf Node {leaf} | No-Show Prob: {prob:.1%} | Decision: {decision}")
