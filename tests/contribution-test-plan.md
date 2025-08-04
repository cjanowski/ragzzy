# RagZzy Contribution Functionality Test Plan

## Overview
This document outlines comprehensive test scenarios for the RagZzy chat application's contribution functionality, focusing on the recent improvements made to text visibility, toggle functionality, and user experience.

## Test Environment Setup
- **Browser**: Chrome, Firefox, Safari, Edge
- **Device Types**: Desktop (1920x1080), Tablet (768x1024), Mobile (375x667)
- **Network Conditions**: Online, Slow 3G, Offline
- **Test Data**: Prepared sample questions and answers of varying lengths

---

## Test Suite 1: Text Visibility Testing

### Test Case 1.1: Form Field Text Visibility (Critical)
**Objective**: Verify all text is clearly visible in contribution modal form fields

**Preconditions**:
- Application loaded successfully
- Contribution toggle is enabled

**Test Steps**:
1. Open RagZzy application
2. Trigger contribution modal via any method:
   - Click "Help Out" button in contribution prompt
   - Use keyboard shortcut Ctrl+K (Cmd+K on Mac)
   - Click guided prompt items
3. Inspect each form field:
   - Question input field
   - Answer textarea
   - Category dropdown
   - Confidence slider
4. Type sample text in each field
5. Verify text color contrast meets WCAG AA standards (4.5:1 ratio)

**Expected Results**:
- All form fields display white background (`#ffffff`)
- All text appears in dark color (`#1f2937`)
- Text is clearly readable without strain
- Placeholder text is visible but distinguishable from user input
- Character counter updates correctly and is visible

**Test Data**:
```
Question: "What are your business hours?"
Answer: "We are open Monday through Friday from 9 AM to 6 PM EST. Weekend hours are Saturday 10 AM to 4 PM. We're closed on Sundays and major holidays."
```

### Test Case 1.2: Text Visibility Across Themes
**Objective**: Ensure text remains visible regardless of browser/system theme

**Test Steps**:
1. Test with browser in light mode
2. Test with browser in dark mode
3. Test with system dark mode enabled
4. Test with high contrast mode enabled

**Expected Results**:
- Form fields maintain white background with `!important` declarations
- Text color remains dark and readable in all theme configurations

---

## Test Suite 2: Contribution Toggle Functionality

### Test Case 2.1: Toggle Switch Operation
**Objective**: Verify the contribution toggle in header works correctly

**Test Steps**:
1. Locate contribution toggle in header (next to settings button)
2. Verify initial state (should be enabled by default)
3. Click toggle to disable contributions
4. Observe success toast message
5. Click toggle to enable contributions
6. Observe success toast message

**Expected Results**:
- Toggle switch animates smoothly between states
- Success toast displays appropriate message:
  - "Contribution prompts enabled" when enabled
  - "Contribution prompts disabled" when disabled
- Visual state correctly reflects the setting

### Test Case 2.2: Toggle Effect on Contribution Prompts
**Objective**: Verify toggle controls visibility of contribution prompts

**Test Steps**:
1. Enable contribution toggle
2. Send a message that would trigger contribution prompt
3. Verify contribution prompt appears
4. Disable contribution toggle
5. Verify existing contribution prompts are hidden
6. Send another message that would trigger contribution prompt
7. Verify no new contribution prompts appear

**Expected Results**:
- When enabled: Contribution prompts appear as expected
- When disabled: All contribution prompts are hidden immediately
- New contribution prompts don't appear when disabled

### Test Case 2.3: Toggle Persistence
**Objective**: Verify toggle preference is saved and restored

**Test Steps**:
1. Set toggle to disabled state
2. Refresh the page (F5)
3. Verify toggle remains disabled
4. Set toggle to enabled state
5. Close and reopen browser tab
6. Verify toggle remains enabled
7. Test in incognito/private browsing mode

**Expected Results**:
- Setting persists across page refreshes
- Setting persists across browser sessions
- Incognito mode uses default setting (enabled)

---

## Test Suite 3: End-to-End Contribution Flow

### Test Case 3.1: Complete Contribution Submission
**Objective**: Test full contribution workflow from trigger to completion

**Test Steps**:
1. Send message: "What is your refund policy?"
2. Wait for bot response and contribution prompt
3. Click "Help Out" button
4. Fill in contribution form:
   - Question: "What is your refund policy?"
   - Answer: "We offer full refunds within 30 days of purchase. Items must be in original condition. Refunds are processed within 5-7 business days."
   - Category: "Policies"
   - Confidence: 5/5
5. Click "Add Knowledge" button
6. Verify success message
7. Check if follow-up suggestions appear

**Expected Results**:
- Modal opens with pre-filled question
- Form validates correctly before submission
- Loading state shows during submission
- Success toast appears: "Thank you for clarifying pricing information - very helpful!"
- Modal closes automatically
- Follow-up suggestions may appear
- Knowledge base file is updated

### Test Case 3.2: Contribution Validation
**Objective**: Test form validation and error handling

**Test Steps**:
1. Open contribution modal
2. Test minimum length validation:
   - Question with 4 characters: "Test"
   - Answer with 9 characters: "Too short"
3. Test maximum length validation:
   - Answer with 2001+ characters
4. Test empty field validation:
   - Leave question empty
   - Leave answer empty
5. Test special characters and HTML injection

**Expected Results**:
- Validation errors appear for invalid inputs:
  - "Please provide a question (at least 5 characters)"
  - "Please provide an answer (at least 10 characters)"
  - "Answer is too long (maximum 2000 characters)"
- Form cannot be submitted with invalid data
- No HTML injection occurs

### Test Case 3.3: Character Counter Functionality
**Objective**: Verify character counter works correctly

**Test Steps**:
1. Open contribution modal
2. Type in answer field and monitor character counter
3. Test color changes:
   - < 1600 characters (gray)
   - 1600-1800 characters (yellow)
   - > 1800 characters (red)
4. Test at exactly 2000 characters

**Expected Results**:
- Counter updates in real-time
- Color changes appropriately based on length
- Counter shows format "X / 2000 characters"

---

## Test Suite 4: Mobile Responsiveness

### Test Case 4.1: Mobile Modal Display
**Objective**: Verify contribution modal works on mobile devices

**Test Steps**:
1. Access RagZzy on mobile device (or browser dev tools mobile view)
2. Trigger contribution modal
3. Test modal sizing and positioning
4. Test form field usability
5. Test keyboard interaction
6. Test scrolling within modal

**Expected Results**:
- Modal fits properly within mobile viewport
- Form fields are appropriately sized (min 48px height)
- Keyboard doesn't obscure form fields
- Modal is scrollable if content exceeds viewport
- Buttons are touch-friendly (min 44px)

### Test Case 4.2: Touch Interactions
**Objective**: Test touch-specific interactions

**Test Steps**:
1. Test toggle switch with touch
2. Test confidence slider with touch
3. Test dropdown selection with touch
4. Test textarea resizing with touch

**Expected Results**:
- All touch targets respond appropriately
- Touch feedback is provided
- No accidental activations occur

---

## Test Suite 5: Advanced Features Testing

### Test Case 5.1: Keyboard Shortcuts
**Objective**: Test keyboard accessibility features

**Test Steps**:
1. Test Ctrl+K (Cmd+K) to open contribution modal
2. Test Escape key to close modal
3. Test Tab navigation through form fields
4. Test Enter key to submit form
5. Test screen reader compatibility

**Expected Results**:
- All keyboard shortcuts work as expected
- Focus management is proper
- Screen readers can navigate form correctly

### Test Case 5.2: API Integration
**Objective**: Test backend integration

**Test Steps**:
1. Submit valid contribution
2. Monitor network requests in browser dev tools
3. Verify API endpoint responses
4. Test error scenarios:
   - Network disconnection
   - Server error (500)
   - Invalid API response

**Expected Results**:
- POST request to `/api/contribute` succeeds
- Proper error handling for failure scenarios
- User receives appropriate feedback

### Test Case 5.3: Knowledge Base Integration
**Objective**: Verify contributions are properly stored

**Test Steps**:
1. Submit a contribution
2. Check `knowledge_base.txt` file for new entry
3. Send a question related to the contribution
4. Verify AI can use the new knowledge

**Expected Results**:
- Contribution is appended to knowledge base file
- Format matches expected structure
- AI responses improve with new knowledge

---

## Test Suite 6: Error Handling & Edge Cases

### Test Case 6.1: Network Error Handling
**Objective**: Test behavior during network issues

**Test Steps**:
1. Start contribution submission
2. Disconnect network during submission
3. Reconnect network
4. Test with slow network (throttled)

**Expected Results**:
- Appropriate error messages displayed
- No data loss occurs
- Graceful degradation

### Test Case 6.2: XSS and Security Testing
**Objective**: Ensure security against malicious input

**Test Steps**:
1. Test XSS payloads in form fields:
   ```
   <script>alert('xss')</script>
   javascript:alert('xss')
   ```
2. Test SQL injection patterns
3. Test very large payloads

**Expected Results**:
- All malicious content is properly sanitized
- No script execution occurs
- Large payloads are handled gracefully

---

## Performance Testing

### Test Case P.1: Load Performance
**Objective**: Verify contribution functionality doesn't impact performance

**Test Steps**:
1. Measure page load time with contribution features
2. Monitor memory usage during modal operations
3. Test with 100+ contributions in knowledge base

**Expected Results**:
- Page load time < 3 seconds
- Modal opens < 500ms
- Memory usage remains stable

---

## Automated Test Implementation

### Test Framework: Playwright + Jest
```javascript
// Example test structure
describe('Contribution Toggle Tests', () => {
  test('should toggle contribution prompts on/off', async () => {
    // Implementation
  });
  
  test('should persist toggle state', async () => {
    // Implementation
  });
});
```

### CI/CD Integration
- Tests run on pull requests
- Cross-browser testing via GitHub Actions
- Mobile testing via BrowserStack integration

---

## Test Data Management

### Sample Test Data
```json
{
  "validContributions": [
    {
      "question": "What are your business hours?",
      "answer": "We are open Monday through Friday from 9 AM to 6 PM EST.",
      "category": "business-info",
      "confidence": 5
    }
  ],
  "invalidContributions": [
    {
      "question": "Hi",
      "answer": "Short",
      "expectedError": "Question must be at least 5 characters long"
    }
  ]
}
```

---

## Success Criteria

### Critical Requirements (Must Pass)
- ✅ All form fields display text clearly
- ✅ Toggle functionality works correctly
- ✅ End-to-end contribution flow completes
- ✅ Mobile responsiveness meets standards
- ✅ Toggle preference persists

### Important Requirements (Should Pass)
- ✅ Keyboard accessibility works
- ✅ Error handling is comprehensive
- ✅ Security measures prevent XSS
- ✅ Performance meets targets

### Nice-to-Have Requirements (Could Pass)
- ✅ Advanced keyboard shortcuts work
- ✅ Screen reader compatibility
- ✅ Offline graceful degradation

---

## Bug Report Template

```markdown
## Bug Report

**Test Case**: [Test Case ID]
**Environment**: [Browser/Device/OS]
**Severity**: [Critical/High/Medium/Low]

**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Result**:

**Actual Result**:

**Screenshots/Videos**:

**Additional Notes**:
```

---

## Test Schedule

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | 2 hours | Text visibility and toggle functionality |
| Phase 2 | 3 hours | End-to-end contribution flow |
| Phase 3 | 2 hours | Mobile responsiveness |
| Phase 4 | 1 hour | Edge cases and security |
| Phase 5 | 1 hour | Performance and accessibility |

**Total Estimated Time**: 9 hours

---

## Conclusion

This comprehensive test plan ensures the RagZzy contribution functionality meets all quality standards and provides a reliable user experience across all supported platforms and scenarios.