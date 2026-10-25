"""Write the fixed, hand-authored synthetic fixture. No model or real inbox is used.

Cases and split membership were authored before the first inference run. Do not
edit labels or split membership in response to model probabilities or test scores.
"""
import json
from pathlib import Path

DEFAULT_HISTORY = "You are the project owner and the direct recipient. No reply has yet been sent."

# (expected label, subject, body, optional thread context)
CASES = {
    "train": [
        (True, "Release sign-off", "Can you approve the release checklist by Thursday?", ""),
        (False, "Release shipped", "The deployment completed successfully. This is an FYI; no action needed.", ""),
        (True, "Budget clarification", "Which budget code should I use for the workshop booking?", ""),
        (False, "Weekly digest", "Here is the automated digest of changes to your subscribed projects.", ""),
        (True, "Meeting time", "Would Tuesday at 10 work for our planning call? Please let me know.", ""),
        (False, "Calendar receipt", "Your RSVP has been recorded. There is nothing further to do.", ""),
        (True, "Contract question", "I am blocked on your answer about whether the renewal includes training.", ""),
        (False, "Contract answered", "Thanks, that answers the renewal question. We are all set.", "You already answered the sender's renewal question."),
        (True, "Small favor", "It would help if you could send me the latest rollout dates before lunch.", ""),
        (False, "Rollout dates", "Sharing the rollout dates for your records. Please do not reply to this automated mailbox.", ""),
        (True, "Draft review", "When you get a chance, I would appreciate your comments on the attached proposal.", ""),
        (False, "Optional reading", "Here is a draft you might enjoy. Comments welcome if you have any, but entirely optional.", ""),
        (True, "Status confirmation", "Are we still expecting your team's migration to finish this week?", ""),
        (False, "Status broadcast", "The migration remains on schedule. This message is informational.", ""),
        (True, "Decision outstanding", "Following up on the choice between the two vendors. We need your decision to proceed.", ""),
        (False, "Decision assigned", "Jordan, could you choose the vendor? Copying you as the project owner for visibility.", "Jordan owns this decision. You are copied for awareness and have no assigned action."),
        (True, "Sanity check", "Could you confirm that the dates below match what you agreed with the customer?", ""),
        (False, "Confirmation received", "We received your confirmation of the dates. Thank you.", "You confirmed the dates yesterday."),
        (True, "Unclear owner", "I think this falls in your area. Who should handle the access review?", ""),
        (False, "Access completed", "The access review finished automatically and requires no response.", ""),
        (True, "Quick check-in", "Any update on the document you said you would send me?", "You promised to send the document today; it has not been sent."),
        (False, "Document archive", "A copy of the document is now in the archive for reference.", ""),
        (True, "Travel plan", "Please tell me whether you will attend in person so I can reserve the room.", ""),
        (False, "Event invitation", "Open office hours are available Friday. Join if useful; no RSVP required.", ""),
        (True, "Gentle reminder", "Just checking whether you had a chance to look at my question below.", "The unanswered question asks you to confirm the project's delivery date."),
        (False, "Reminder canceled", "Please ignore my earlier request. I found the answer and need no follow-up.", "The earlier thread asked you for a delivery date."),
        (True, "Numbers need checking", "These totals look different from your estimate. Could you explain the difference?", ""),
        (False, "Receipt", "Payment received. This automated receipt is for your records.", ""),
        (True, "Thoughts on approach", "I would value your recommendation on which approach to take before we start.", "You are responsible for selecting the implementation approach."),
        (False, "Industry newsletter", "This month's newsletter covers industry news and product announcements.", ""),
        (True, "If feasible", "If feasible, please let me know by tomorrow whether your team can cover the weekend.", "You coordinate staffing and must respond even if coverage is unavailable."),
        (False, "Weekend covered", "Staffing is confirmed and there are no open questions for your team.", ""),
        (True, "Reopened item", "The previous fix did not hold. Can you suggest the next step?", "You replied yesterday, but the sender now reports a new failure."),
        (False, "Forwarded for awareness", "Forwarding the thread so you are aware of the discussion.", "All requests in the forwarded thread are assigned to another team; you have no action."),
        (True, "Acknowledgment needed", "Please acknowledge you received this revised schedule, even if you have no comments.", ""),
        (False, "No acknowledgment", "Here is the revised schedule. No acknowledgment or feedback is required.", ""),
    ],
    "validation": [
        (True, "Launch dependency", "We cannot finalize launch until you confirm the support contact. Who should we list?", ""),
        (False, "Build notice", "Build 604 passed all checks. This is an automatic notification.", ""),
        (True, "A suggestion needed", "I am unsure how to explain the delay. What would you recommend telling the customer?", ""),
        (False, "Suggestion resolved", "Your advice worked. The customer accepted the updated plan; nothing else needed.", "You already provided the requested advice."),
        (True, "Room capacity", "Do you expect more than twelve people? I need your answer to book a suitable room.", ""),
        (False, "Room reserved", "The room is reserved. This receipt does not require confirmation.", ""),
        (True, "No rush", "No rush, but could you send your updated contact list when you have a moment?", ""),
        (False, "Optional feedback", "We published a survey if you feel like sharing feedback. Participation is optional.", ""),
        (True, "Still waiting", "Circling back on the item below; your response would help us unblock planning.", "The open item is your approval of the revised scope."),
        (False, "Scope approved elsewhere", "The sponsor approved the scope. Please disregard my earlier approval request.", "The sender previously asked you to approve the scope."),
        (True, "Ownership confirmation", "Please confirm whether you own the dashboard or point me to the correct owner.", ""),
        (False, "Ownership directory", "Sharing the current ownership directory for reference only.", ""),
        (True, "One detail", "Would you mind checking whether the customer name in this draft is correct?", ""),
        (False, "Handled by colleague", "Taylor, can you check the customer name? Adding you for visibility.", "Taylor is responsible for checking the name; you have no assigned action."),
        (True, "Date changed", "The original date no longer works. Can you propose another day?", "You agreed to the original date last week; this is a new request."),
        (False, "Date accepted", "The new date is fine with everyone. No further reply needed.", ""),
        (True, "Final document", "Whenever you can, please return the final document with your signature.", ""),
        (False, "Document downloaded", "Your document download completed. Do not reply to this system message.", ""),
    ],
    "test": [
        (True, "Customer handoff", "Before I transfer the account, could you confirm who will run onboarding?", ""),
        (False, "Storage alert resolved", "The storage warning cleared automatically. You do not need to respond.", ""),
        (True, "Capacity question", "Can your team take two more tasks next week? A yes or no will help us plan.", ""),
        (False, "Task report", "Attached is the completed-task report for your records.", ""),
        (True, "Would appreciate input", "I would appreciate your assessment of the delivery risk before I brief leadership.", "You are the person responsible for assessing this delivery risk."),
        (False, "Risk briefing complete", "The briefing is over and all questions have been answered. Sharing the notes as an FYI.", ""),
        (True, "Quiet follow-up", "Checking in again about the request I sent on Monday.", "Monday's message asks you to supply the missing purchase order number; no answer was sent."),
        (False, "Found the number", "I located the purchase order number. You can ignore the question from Monday.", ""),
        (True, "Choice for demo", "Should we demonstrate the new workflow or the existing one? Please choose so I can prepare.", ""),
        (False, "Demo recording", "Here is the demo recording. Watch whenever useful; no response expected.", ""),
        (True, "Late request", "Sorry for the short notice. Please confirm your availability for tomorrow's call.", ""),
        (False, "Call canceled", "Tomorrow's call is canceled and your availability confirmation is no longer needed.", ""),
        (True, "Second attempt", "The login instructions you sent did not work. What should I try next?", "You answered the first request, but this follow-up remains unanswered."),
        (False, "Question for Sam", "Sam, could you send the login instructions? Copying you so you know we are following up.", "Sam owns the response. You are copied and do not need to reply."),
        (True, "Approval by exception", "If you cannot approve this expense, please explain why. Otherwise send your approval.", "You are the required expense approver."),
        (False, "Invitation to browse", "The updated knowledge base is available. Browse it if interested; no action required.", ""),
        (True, "Need a receipt confirmation", "Please reply to confirm you received the revised statement of work.", ""),
        (False, "Confirmation complete", "We logged your acknowledgment of the statement of work. This completes the exchange.", "You already acknowledged the revised statement of work."),
    ],
}


def main():
    rows = []
    for split, cases in CASES.items():
        for index, (label, subject, body, history) in enumerate(cases):
            rows.append({
                "id": f"{split}-{index + 1:03d}", "split": split,
                "sender": f"Colleague {index + 1} <colleague{index + 1}@example.invalid>",
                "subject": subject, "body": body, "history": history or DEFAULT_HISTORY,
                # Alternate tier independently of the alternating label.
                "customer_tier": "critical" if (index // 2) % 2 == 0 else "standard",
                "needs_response": label,
                "label_rationale": "Outstanding request assigned to the recipient." if label else
                                   "Informational, optional, resolved, automated, or assigned to someone else.",
            })
    target = Path(__file__).resolve().parents[1] / "data" / "synthetic_emails.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} hand-authored synthetic emails to {target}")


if __name__ == "__main__":
    main()
